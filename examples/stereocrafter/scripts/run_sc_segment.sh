#!/bin/bash
# StereoCrafter 单段处理 (Step1-4, 不含 autorotate)
# 由 run_sc_parallel.sh 调用: _auto.mp4 已切好段, 每段独立跑 4 步, 最后拼接
# ==================================================
# 用法: ./run_sc_segment.sh <seg.mp4> --out <dir> --gpu <N> [--max-disp <px>] [--tile <N>]
# 输入: seg.mp4 = 已 autorotate + 切片的视频段 (1080x1920, 带音频)
# 输出: {out}/final_sbs.mp4 + final_sbs_SWAPPED.mp4
# ==================================================

set -e
IN_VIDEO="${1:?用法: $0 <seg.mp4> --out <dir> --gpu <N> [选项]}"
shift
MAXDISP=20; TILE=2; GPU=0; OUT=./seg_out
while [ $# -gt 0 ]; do
  case "$1" in
    --max-disp) MAXDISP="$2"; shift 2;;
    --tile)     TILE="$2"; shift 2;;
    --gpu)      GPU="$2"; shift 2;;
    --out)      OUT="$2"; shift 2;;
    *) echo "未知选项: $1"; exit 1;;
  esac
done
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
W="${SC_WEIGHTS:-$SC/weights}"
mkdir -p "$OUT"
cd "$SC"

BASE=$(basename "$IN_VIDEO" .mp4)

# ---- Step 1: DepthCrafter 深度 + forward-warp splatting ----
CUDA_VISIBLE_DEVICES=$GPU python depth_splatting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/DepthCrafter \
  --input_video_path "$IN_VIDEO" \
  --output_video_path $OUT/splatting_results.mp4 \
  --max_disp $MAXDISP \
  > $OUT/stage1.log 2>&1
echo "[seg] Step1 depth+splatting done (max_disp=$MAXDISP)"

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
