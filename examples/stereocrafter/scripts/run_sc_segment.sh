#!/bin/bash
# StereoCrafter 单段处理 (Step1-4, 不含 autorotate)
# 由 run_sc_parallel.sh 调用: _auto.mp4 已切好段, 每段独立跑 4 步, 最后拼接
# ==================================================
# 用法: ./run_sc_segment.sh <seg.mp4> --out <dir> --gpu <N> [--max-disp <px>] [--tile <N>]
#       [--auto-video <path> --frame-start <N> --frame-end <N>]
#   --auto-video: 全量 _auto.mp4, depth 退化时用于上下文扩展重跑
#   --frame-start/end: 本段在 auto.mp4 中的帧范围(0-indexed, end 不含)
# 输入: seg.mp4 = 已 autorotate + 切片的视频段 (1080x1920, 带音频)
# 输出: {out}/final_sbs.mp4 + final_sbs_SWAPPED.mp4
# ==================================================

set -e
IN_VIDEO="${1:?用法: $0 <seg.mp4> --out <dir> --gpu <N> [选项]}"
shift
MAXDISP=20; TILE=2; GPU=0; OUT=./seg_out
AUTO_VIDEO=""; FRAME_START=""; FRAME_END=""
while [ $# -gt 0 ]; do
  case "$1" in
    --max-disp)    MAXDISP="$2"; shift 2;;
    --tile)        TILE="$2"; shift 2;;
    --gpu)         GPU="$2"; shift 2;;
    --out)         OUT="$2"; shift 2;;
    --auto-video)  AUTO_VIDEO="$2"; shift 2;;
    --frame-start) FRAME_START="$2"; shift 2;;
    --frame-end)   FRAME_END="$2"; shift 2;;
    *) echo "未知选项: $1"; exit 1;;
  esac
done
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
W="${SC_WEIGHTS:-$SC/weights}"
mkdir -p "$OUT"
cd "$SC"

BASE=$(basename "$IN_VIDEO" .mp4)

# ---- Step 1: DepthCrafter 深度 + forward-warp splatting (OOM retry) ----
# DepthCrafter 并发峰值偶发 OOM: max_split_size_mb=128 减碎片 + 失败重跑 (GPU 释放后该成功)
# --save_depth=True: 落盘 splatting_results.npz (depth 原数组) + _depth_vis.mp4,
# 配合 --keep 保留中间产物, 便于 depth 退化/OOD 诊断
for _att in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    python depth_splatting_inference.py \
    --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path $W/DepthCrafter \
    --input_video_path "$IN_VIDEO" \
    --output_video_path $OUT/splatting_results.mp4 \
    --max_disp $MAXDISP \
    --save_depth=True \
    > $OUT/stage1.log 2>&1 && break
  [ "$_att" = 3 ] && { echo "[seg] Step1 attempt $_att 失败: $(grep -m1 OutOfMemory $OUT/stage1.log 2>/dev/null || head -1 $OUT/stage1.log)"; exit 1; }
  echo "[seg] Step1 attempt $_att 失败 (OOM?), 清显存重跑..."
  sleep 5
done
echo "[seg] Step1 depth+splatting done (max_disp=$MAXDISP)"

# ---- Step 1b: 上下文扩展重试 (depth 退化时, gradient 兜底不足以防黑) ----
# DepthCrafter 对低纹理/极暗帧序列产全 NaN (fp16 下溢), 例: seg2 帧250-399
# 独占 GPU/换 GPU/重跑均无效 (内容决定), 但混合进正常帧后深度正常
# (example_noseg 686帧全量 0 退化)。故: 退化时从 auto.mp4 取前后 CONTEXT_FRAMES
# 帧垫入, 重跑深度取原帧范围, splatting grid 裁回 — 模拟"不分段"的稀释效果。
CONTEXT_FRAMES=75   # 每侧扩展帧数 (example: seg2 150帧+75*2=300帧, 稀释比 2x)
if grep -q "重跑仍退化" $OUT/stage1.log 2>/dev/null; then
  if [ -n "$AUTO_VIDEO" ] && [ -n "$FRAME_START" ] && [ -n "$FRAME_END" ] && [ -f "$AUTO_VIDEO" ]; then
    echo "[seg] depth 退化, 上下文扩展重试 (auto=$AUTO_VIDEO frames [$FRAME_START:$FRAME_END))"
    EXPAND_S=$(( FRAME_START > CONTEXT_FRAMES ? FRAME_START - CONTEXT_FRAMES : 0 ))
    TOTAL_FR=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$AUTO_VIDEO" 2>/dev/null)
    TOTAL_FR=${TOTAL_FR//[^0-9]/}; TOTAL_FR=${TOTAL_FR:-0}
    EXPAND_E=$(( FRAME_END + CONTEXT_FRAMES ))
    [ "$EXPAND_E" -gt "$TOTAL_FR" ] && EXPAND_E=$TOTAL_FR
    EXPAND_CTX="$OUT/seg_expanded.mp4"
    echo "[seg] 扩展范围 [$EXPAND_S:$EXPAND_E), 原段 [$FRAME_START:$FRAME_END)"
    ffmpeg -y -v error -i "$AUTO_VIDEO" \
      -vf "select=between(n\,${EXPAND_S}\,$((EXPAND_E-1))),setpts=PTS-STARTPTS" \
      -c:v libx264 -crf 18 -an "$EXPAND_CTX"
    EXPAND_OUT="$OUT/splatting_expanded.mp4"
    for _att in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python depth_splatting_inference.py \
        --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
        --unet_path $W/DepthCrafter \
        --input_video_path "$EXPAND_CTX" \
        --output_video_path "$EXPAND_OUT" \
        --max_disp $MAXDISP \
        > $OUT/stage1_expanded.log 2>&1 && break
      [ "$_att" = 3 ] && { echo "[seg] 扩展重跑 Step1 3次均失败, 放弃, 沿用原(可能黑帧)"; }
      sleep 5
    done
    if [ -f "$EXPAND_OUT" ] && ! grep -q "重跑仍退化" $OUT/stage1_expanded.log 2>/dev/null; then
      # 从扩展 splatting grid 中裁出原段帧范围 (select 帧索引)
      CROP_START=$(( FRAME_START - EXPAND_S ))   # 原段在扩展 clip 中的起始帧
      SEG_LEN=$(( FRAME_END - FRAME_START ))
      CROP_END=$(( CROP_START + SEG_LEN - 1 ))
      echo "[seg] 扩展深度正常, 裁 splatting grid 帧 [$CROP_START:$CROP_END] → 替换原 splatting_results.mp4"
      ffmpeg -y -v error -i "$EXPAND_OUT" \
        -vf "select='between(n\,${CROP_START}\,${CROP_END})',setpts=PTS-STARTPTS" \
        -c:v libx264 -crf 16 -an "$OUT/splatting_results.mp4"
    else
      echo "[seg] 扩展重跑仍退化, 保留原 gradient 兜底 splatting_results.mp4 (右眼可能黑)"
    fi
    rm -f "$EXPAND_CTX" "$EXPAND_OUT" $OUT/stage1_expanded.log 2>/dev/null
  else
    echo "[seg] depth 退化但缺少 --auto-video/--frame-start/--frame-end, 保留原兜底 splatting"
  fi
fi

# ---- 中间产物拆存: grid 四象限裁成单独 mp4, 便于直接查看 (配合 --keep) ----
# splatting_results.mp4 2×2: TL=左眼原图 TR=depth_vis BL=occlu_mask BR=warped右眼
GW=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 $OUT/splatting_results.mp4)
GH=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $OUT/splatting_results.mp4)
GW2=$((GW/2)); GH2=$((GH/2))
ffmpeg -y -v error -i $OUT/splatting_results.mp4 -vf "crop=${GW2}:${GH2}:${GW2}:0"     -c:v libx264 -crf 16 $OUT/debug_depth_vis.mp4
ffmpeg -y -v error -i $OUT/splatting_results.mp4 -vf "crop=${GW2}:${GH2}:0:${GH2}"    -c:v libx264 -crf 16 $OUT/debug_occlu_mask.mp4
ffmpeg -y -v error -i $OUT/splatting_results.mp4 -vf "crop=${GW2}:${GH2}:${GW2}:${GH2}" -c:v libx264 -crf 16 $OUT/debug_warped_right.mp4
echo "[seg] 中间产物拆存: debug_depth_vis/debug_occlu_mask/debug_warped_right.mp4"

# ---- Step 2: 右眼 inpainting (tile_num=2, 竖屏quadrant 1024x1920) ----
CUDA_VISIBLE_DEVICES=$GPU python inpainting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/StereoCrafter \
  --input_video_path $OUT/splatting_results.mp4 \
  --save_dir $OUT \
  --tile_num $TILE \
  >> $OUT/stage1.log 2>&1
echo "[seg] Step2 right-eye inpainting done (tile=$TILE)"

# ---- Step 3: 左眼也过SVD(两眼协调) ----
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$IN_VIDEO")
W2=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$IN_VIDEO")
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$IN_VIDEO")
ffmpeg -y -v error -i $OUT/splatting_results.mp4 -vf "crop=${W2}:${H}:0:0" $OUT/left_eye.mp4
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $OUT/left_eye.mp4)
ffmpeg -y -v error -i $OUT/left_eye.mp4 \
  -f lavfi -i color=black:s=${W2}x${H}:r=$FPS:d=$DUR \
  -f lavfi -i color=white:s=${W2}x${H}:r=$FPS:d=$DUR \
  -i $OUT/left_eye.mp4 \
  -filter_complex "[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]" \
  -map "[v]" -c:v libx264 -crf 16 -pix_fmt yuv420p $OUT/left_pass_grid.mp4
CUDA_VISIBLE_DEVICES=$GPU python inpainting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/StereoCrafter \
  --input_video_path $OUT/left_pass_grid.mp4 \
  --save_dir $OUT \
  --tile_num $TILE \
  >> $OUT/stage1.log 2>&1
echo "[seg] Step3 left-eye SVD pass done"

# ---- Step 4: 拼 SBS = [SVD左 | SVD右] (段内无音频, 音频在 parallel 拼接时合) ----
RH=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $OUT/splatting_results_inpainting_results_sbs.mp4)
RW=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $OUT/splatting_results_inpainting_results_sbs.mp4)
HW=$((RW/2))
ffmpeg -y -v error -i $OUT/splatting_results_inpainting_results_sbs.mp4 \
  -vf "crop=${HW}:${RH}:${HW}:0" $OUT/svd_right.mp4
ffmpeg -y -v error -i $OUT/left_pass_grid_inpainting_results_sbs.mp4 \
  -vf "crop=${HW}:${RH}:${HW}:0" $OUT/svd_left.mp4
# 只生成 final_sbs (无 SWAPPED, SWAPPED 在 parallel 脚本最后统一做)
ffmpeg -y -v error -i $OUT/svd_left.mp4 -i $OUT/svd_right.mp4 \
  -filter_complex "[0:v][1:v]hstack=2[v]" -map "[v]" \
  -c:v libx264 -crf 16 -pix_fmt yuv420p $OUT/final_sbs.mp4
echo "[seg] Step4 final SBS: $OUT/final_sbs.mp4"
echo "SEG_DONE"
