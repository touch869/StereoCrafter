#!/bin/bash
# StereoCrafter 分段 + 多卡并行 (竖屏 portrait)
# ==================================================
# 把长视频切成 N 段(带 overlap), 每段在一张卡上独立跑 4 步(SC pipeline),
# 最后按段顺序 concat 去重(overlap 区取前段尾部) + 合源音频。
#
# 解决: 0.mp4 (2699帧/90s) Step1 DepthCrafter 一次性全量帧上 GPU → 24G 卡 OOM。
# 分段后每段帧张量小, 单卡不爆; 多卡并行 wall-clock ≈ 单段时长。
#
# 用法:
#   ./run_sc_parallel.sh <input.mp4> --out <dir> --gpus <0,1,2,3,4> \
#     [--chunk-frames 600] [--overlap 25] [--max-disp 40] [--tile 2]
#   默认: chunk-frames=600, overlap=25, max-disp=20, tile=2
#   --gpus 逗号分隔的 GPU id 列表; 段数 = len(gpus), 每段帧数自适应
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

# ---- 切片: 按 chunk-frames + overlap 切成 NUM_GPUS 段 ----
TOTAL_FR=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$AUTO" 2>/dev/null | tail -1)
TOTAL_FR=${TOTAL_FR//[^0-9]/}
# count_frames 慢; 退化用 nb_frames
if [ -z "$TOTAL_FR" ] || [ "$TOTAL_FR" = "0" ]; then
  TOTAL_FR=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$AUTO" 2>/dev/null)
  TOTAL_FR=${TOTAL_FR//[^0-9]/}
fi
echo "[0] total frames = $TOTAL_FR"

# 每段帧数 = CHUNK, 段间 overlap = OVERLAP; 实际段数由 NUM_GPUS 决定
# 段 i 的帧范围: [i*(CHUNK-OVERLAP), i*(CHUNK-OVERLAP)+CHUNK)
STRIDE=$((CHUNK - OVERLAP))
echo "[0] chunk=$CHUNK overlap=$OVERLAP stride=$STRIDE gpus=$GPUS"

SEG_FILES=()
PIDS=()
for i in "${!GPU_ARR[@]}"; do
  G=${GPU_ARR[$i]}
  START_FR=$((i * STRIDE))
  END_FR=$((START_FR + CHUNK))
  if [ "$START_FR" -ge "$TOTAL_FR" ]; then break; fi
  # 最后一段不超过总帧数
  [ "$END_FR" -gt "$TOTAL_FR" ] && END_FR=$TOTAL_FR

  SEG_DIR="$OUT/seg${i}"
  SEG_MP4="$SEG_DIR/seg${i}.mp4"
  SEG_FILES+=("$SEG_DIR/final_sbs.mp4")
  mkdir -p "$SEG_DIR"

  if [ -f "$SEG_DIR/final_sbs.mp4" ]; then
    echo "[seg$i] RESUME: final_sbs.mp4 已存在, 跳过 (gpu=$G)"
    continue
  fi

  # 切片 (帧精确: -vf select+setpts; 段内不需要音频, 最后拼接再合源音)
  echo "[seg$i] 切片 frames [$START_FR:$END_FR] → gpu=$G"
  ffmpeg -y -v error -i "$AUTO" \
    -vf "select=between(n\,${START_FR}\,$((END_FR-1))),setpts=PTS-STARTPTS" \
    -c:v libx264 -crf 18 -an "$SEG_MP4"

  # 并行启动该段 (后台)
  echo "[seg$i] 启动 SC pipeline (gpu=$G)"
  bash "$SCRIPT_DIR/run_sc_segment.sh" "$SEG_MP4" \
    --out "$SEG_DIR" --gpu "$G" --max-disp "$MAXDISP" --tile "$TILE" \
    > "$SEG_DIR/seg.log" 2>&1 &
  PIDS+=($!)
done

# ---- 等所有段完成 ----
echo "[parallel] 等待 ${#PIDS[@]} 段完成..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[parallel] ERROR: 段进程 $pid 失败"
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
      -c:v libx264 -crf 16 -an "$SEG_USE" 2>/dev/null
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
