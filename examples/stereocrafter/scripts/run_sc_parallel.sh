#!/bin/bash
# StereoCrafter 分段 + 多卡工作池 (竖屏 portrait)
# ==================================================
# 把长视频切成 N 段(带 overlap), 每段独立跑 4 步(SC pipeline),
# 最后按段顺序 concat 去重(overlap 区取前段尾部) + 合源音频。
#
# 解决: 0.mp4 (2699帧/90s) Step1 DepthCrafter 一次性全量帧上 GPU → 24G 卡 OOM。
# 分段后单段不爆; 多卡并行 wall-clock ≈ ceil(段数/卡数) × 单段时长。
#
# 段数与卡数解耦 (worker-pool):
#   - 段数 NSEG = 全覆盖所需 (ceil(TOTAL/stride)), 与卡数无关
#   - 段轮流(round-robin)分配到各卡, 每卡串行跑分到的段, 全部卡并行
#   - 段数 > 卡数: 多出的段在各卡排队复用 (不漏覆盖, 修复旧版尾部漏段 bug)
#   - 段数 < 卡数: 多出的卡空闲 (退化回 1段/卡 全并行)
# 注: chunk 是并行粒度参数, 不是显存旋钮 (DepthCrafter 内部 window_size=70
#     分窗, 段长不驱动峰值显存; 显存由 window_size + 1080×1920 降分辨率兜)。
#
# 用法:
#   ./run_sc_parallel.sh <input.mp4> --out <dir> --gpus <0,1,2,3,4> \
#     [--chunk-frames 600] [--overlap 25] [--max-disp 40] [--tile 2]
#   默认: chunk-frames=600, overlap=25, max-disp=20, tile=2
#   --gpus 逗号分隔的 GPU id 列表; 段数自适应(全覆盖), 卡数=并行度
# ==================================================

set -e
INPUT="${1:?用法: $0 <input.mp4> --out <dir> --gpus <0,1,2,...> [选项]}"
shift
OUT=./sc_parallel; GPUS="0"; CHUNK=600; OVERLAP=25; MAXDISP=20; TILE=2
while [ $# -gt 0 ]; do
  case "$1" in
    --out)           OUT="$2"; shift 2;;
    --gpus)          GPUS="$2"; shift 2;;
    --chunk-frames)  CHUNK="$2"; shift 2;;
    --overlap)       OVERLAP="$2"; shift 2;;
    --max-disp)      MAXDISP="$2"; shift 2;;
    --tile)          TILE="$2"; shift 2;;
    *) echo "未知选项: $1"; exit 1;;
  esac
done

SC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
W="${SC_WEIGHTS:-$SC/weights}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
cd "$SC"

# IFS 分隔 GPU 列表
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}

# ---- Step 0: autorotate → _auto.mp4 (1080x1920, -c:a copy 保音频) ----
AUTO="$OUT/_auto.mp4"
if [ -f "$AUTO" ]; then
  echo "[0] RESUME: $AUTO 已存在, 跳过 autorotate"
else
  echo "[0] autorotate + scale → 1080x1920 (保音频)"
  ffmpeg -y -v error -i "$INPUT" -vf "scale=1080:1920" -c:v libx264 -crf 18 -c:a copy "$AUTO"
fi
echo "[0] _auto.mp4: $(ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$AUTO")"
# 源 fps: 切片 + concat trim 都 -r 保持, 防 fps 膨胀 (libx264 默认 25fps 重采样
# 致 20fps 源 150帧→186帧膨胀, 各段帧数/fps 异常 → concat 错位黑段)
AUTO_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$AUTO" 2>/dev/null)
echo "[0] _auto fps=$AUTO_FPS (切片+trim 保持)"

# ---- 切片: 按 chunk-frames + overlap 切成 NUM_GPUS 段 ----
TOTAL_FR=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$AUTO" 2>/dev/null | tail -1)
TOTAL_FR=${TOTAL_FR//[^0-9]/}
# count_frames 慢; 退化用 nb_frames
if [ -z "$TOTAL_FR" ] || [ "$TOTAL_FR" = "0" ]; then
  TOTAL_FR=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$AUTO" 2>/dev/null)
  TOTAL_FR=${TOTAL_FR//[^0-9]/}
fi
echo "[0] total frames = $TOTAL_FR"

# 每段帧数 = CHUNK, 段间 overlap = OVERLAP, stride = CHUNK - OVERLAP
# 段数 NSEG = 全覆盖所需 (与卡数无关); 卡数 = 并行度
STRIDE=$((CHUNK - OVERLAP))
if [ "$STRIDE" -le 0 ]; then
  echo "ERROR: chunk-frames($CHUNK) 必须 > overlap($OVERLAP) (stride=$STRIDE ≤ 0)"; exit 1
fi
echo "[0] chunk=$CHUNK overlap=$OVERLAP stride=$STRIDE gpus=$GPUS"

# ---- Phase A: 生成全部段 (全覆盖, 段数不受卡数限制) ----
SEG_START=()   # 每段起始帧
SEG_END=()     # 每段结束帧 (不含)
SEG_FILES=()   # 每段 final_sbs 路径, 按段顺序 (供 concat, 与 GPU 分配无关)
i=0
while :; do
  S=$((i * STRIDE))
  [ "$S" -ge "$TOTAL_FR" ] && break
  E=$((S + CHUNK))
  [ "$E" -gt "$TOTAL_FR" ] && E=$TOTAL_FR
  SEG_START+=("$S"); SEG_END+=("$E")
  SEG_FILES+=("$OUT/seg${i}/final_sbs.mp4")
  # 本段已覆盖到末尾 → 停, 不切冗余短末段 (seg3 break 修复保留)
  [ "$E" -ge "$TOTAL_FR" ] && break
  i=$((i + 1))
done
NSEG=${#SEG_START[@]}
echo "[0] 段数=$NSEG (全覆盖) on $NUM_GPUS 卡: ${GPU_ARR[*]} → 每卡约 $(( (NSEG + NUM_GPUS - 1) / NUM_GPUS )) 段"
[ "$NSEG" -lt 1 ] && { echo "ERROR: 视频帧数为 0, 无段可切"; exit 1; }

# ---- Phase B: 切片 (全部段, resume 已存在跳过) ----
for ((i=0; i<NSEG; i++)); do
  SEG_DIR="$OUT/seg${i}"
  SEG_MP4="$SEG_DIR/seg${i}.mp4"
  mkdir -p "$SEG_DIR"
  if [ -f "$SEG_DIR/final_sbs.mp4" ]; then
    echo "[seg$i] RESUME: final_sbs.mp4 已存在, 跳过切片+处理"
    continue
  fi
  if [ ! -f "$SEG_MP4" ]; then
    # 帧精确切片 (-vf select+setpts); 段内不要音频, 最后拼接再合源音
    echo "[seg$i] 切片 frames [${SEG_START[$i]}:${SEG_END[$i]}]"
    ffmpeg -y -v error -i "$AUTO" \
      -vf "select=between(n\,${SEG_START[$i]}\,$((${SEG_END[$i]}-1))),setpts=PTS-STARTPTS" \
      -r "$AUTO_FPS" -c:v libx264 -crf 18 -an "$SEG_MP4"
  fi
done

# ---- Phase C: GPU 工作池 (段轮流分配, 每卡串行, 全卡并行) ----
# 段数 > 卡数: 多出的段在各卡排队 (同卡串行) → 不漏覆盖; 段数 < 卡数: 多出的卡空闲
run_one_segment() {   # $1=段idx  $2=真实GPU id
  local idx="$1" gpu="$2"
  local seg_dir="$OUT/seg${idx}"
  [ -f "$seg_dir/final_sbs.mp4" ] && { echo "[seg$idx] RESUME skip (gpu=$gpu)"; return 0; }
  echo "[seg$idx] 启动 SC pipeline (gpu=$gpu)"
  bash "$SCRIPT_DIR/run_sc_segment.sh" "$seg_dir/seg${idx}.mp4" \
    --out "$seg_dir" --gpu "$gpu" --max-disp "$MAXDISP" --tile "$TILE" \
    > "$seg_dir/seg.log" 2>&1
}
gpu_worker() {        # $1=真实GPU id  剩余 $@=该卡要跑的段idx列表
  local gpu="$1"; shift
  for idx in "$@"; do
    if ! run_one_segment "$idx" "$gpu"; then
      echo "[gpu=$gpu] ERROR: seg$idx 失败 (见 $OUT/seg${idx}/seg.log)"
      return 1
    fi
  done
  return 0
}

# 轮流(round-robin)分配: 卡 g 负责 seg{g, g+NUM_GPUS, g+2*NUM_GPUS, ...}
PIDS=()
for g in "${!GPU_ARR[@]}"; do
  real_gpu=${GPU_ARR[$g]}
  assign=()
  for ((j=g; j<NSEG; j+=NUM_GPUS)); do assign+=("$j"); done
  [ ${#assign[@]} -eq 0 ] && continue
  echo "[gpu=$real_gpu] 分配 ${#assign[@]} 段: ${assign[*]}"
  gpu_worker "$real_gpu" "${assign[@]}" &
  PIDS+=($!)
done

# ---- 等所有 GPU 工作进程完成 ----
echo "[parallel] 等待 ${#PIDS[@]} 个 GPU 工作进程 (共 $NSEG 段)..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[parallel] ERROR: 工作进程 $pid 失败"
    FAIL=1
  fi
done
[ "$FAIL" = 1 ] && { echo "ERROR: 至少一段失败, 详见各 seg/seg.log"; exit 1; }

# ---- 拼接: concat 各段 final_sbs, overlap 区去重(取前段尾部) ----
# 每段 final_sbs 帧数 = 该段 seg.mp4 帧数; overlap 区 = 段尾部 OVERLAP 帧
# 去重: 段 i 去掉前 OVERLAP 帧(除了 seg0), concat 剩余
echo "[concat] 拼接各段 final_sbs (overlap=$OVERLAP 去重)"

CONCAT_LIST="$OUT/concat.txt"
> "$CONCAT_LIST"
for i in "${!SEG_FILES[@]}"; do
  SEG_FBS="${SEG_FILES[$i]}"
  if [ ! -f "$SEG_FBS" ]; then
    echo "ERROR: $SEG_FBS 不存在"; exit 1
  fi
  if [ "$i" = 0 ]; then
    # 第一段: 全保留
    SEG_USE="$SEG_FBS"
  else
    # 后续段: 去掉前 OVERLAP 帧 (段内无音频, 只切视频)
    SEG_USE="$OUT/seg${i}_trimmed.mp4"
    ffmpeg -y -v error -i "$SEG_FBS" \
      -vf "select='gte(n\,${OVERLAP})',setpts=PTS-STARTPTS" \
      -r "$AUTO_FPS" -c:v libx264 -crf 16 -an "$SEG_USE" 2>/dev/null
  fi
  echo "file '$SEG_USE'" >> "$CONCAT_LIST"
done

# concat 视频段 (段内无音频, 最后从 _auto.mp4 合源音频)
ffmpeg -y -v error -f concat -safe 0 -i "$CONCAT_LIST" \
  -c:v libx264 -crf 16 -an "$OUT/final_sbs_video.mp4" 2>/dev/null || {
  echo "[concat] 重编码..."
  ffmpeg -y -v error -f concat -safe 0 -i "$CONCAT_LIST" \
    -c:v libx264 -crf 16 -an "$OUT/final_sbs_video.mp4"
}

# 合源音频 (_auto.mp4 带音, Step0 -c:a copy)
ffmpeg -y -v error -i "$OUT/final_sbs_video.mp4" -i "$AUTO" \
  -map 0:v -map 1:a? -c:v copy -c:a aac -shortest "$OUT/final_sbs.mp4"

# SWAPPED 版 (左右眼交换) — 修: 漏了 -map 0:a 导致无声
ffmpeg -y -v error -i "$OUT/final_sbs.mp4" \
  -filter_complex "[0:v]crop=iw/2:ih:0:0[l];[0:v]crop=iw/2:ih:iw/2:0[r];[r][l]hstack[v]" \
  -map "[v]" -map 0:a? -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac "$OUT/final_sbs_SWAPPED.mp4"

echo ""
echo "ALL DONE → $OUT/final_sbs.mp4 (+_SWAPPED)"
ffprobe -v error -show_entries stream=width,height,duration -of default=nw=1 "$OUT/final_sbs.mp4"
