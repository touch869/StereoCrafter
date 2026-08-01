#!/bin/bash
# StereoCrafter 定档生产脚本(横屏): SC04(1080p + tile_num=2) + 左眼SVD + 拼SBS=[SVD左|SVD右]。
# 支持: 时间切片、自动转正、scale到1080p、平移量(--max-disp)、GPU 选择。
#
# 流程: 切片+转正+scale → DepthCrafter深度+forward-warp splatting → 右眼SVD inpainting(tile)
#       → 左眼也过SVD(两眼协调) → 拼 final_sbs.mp4 = [SVD左 | SVD右] + SWAPPED
#
# 用法:
#   ./run_sc_horizontal.sh <input.mp4> [outdir] [选项]
#   选项:
#     --start <sec>  --end <sec>     时间切片 (秒)
#     --width <px> --height <px>     stage1 输入分辨率 (默认 1920x1080)
#     --max-disp <px>                平移量 (默认 20; 实际块视差峰值~-10px, 线性)
#     --tile <N>                     inpainting tile (默认 2; 每tile≈SVD原生1024x576)
#     --gpu <N>                      用第 N 张卡 (默认 0)
#
# 注意:
#   - 4K 输入直接跑会 OOM(splatting 输出 3840x2160 网格), 脚本自动 scale 到 1920x1080。
#   - 竖屏素材请用 run_sc_d_portrait.sh(会自动 autorotate 成 1080x1920)。
#   - 左眼SVD: 两眼都经 SVD 生成, 清晰度/色彩协调(横评定档做法)。
#
# 可移植性(不绑定绝对路径/素材):
#   - 仓库根由脚本位置推导(../..); 权重目录: 环境变量 SC_WEIGHTS(默认 <仓库>/weights)
#   - 请在装有 StereoCrafter 依赖的 env 里运行(python 走 PATH), 不要硬编码 conda 路径
set -e

INPUT="${1:?用法: $0 <input.mp4> [outdir] [选项]}"
shift
OUT=./sc_output
WIDTH=1920; HEIGHT=1080
MAXDISP=20; TILE=2; GPU=0
START=""; END=""

while [ $# -gt 0 ]; do
  case "$1" in
    --start) START="$2"; shift 2;;
    --end)   END="$2"; shift 2;;
    --width) WIDTH="$2"; shift 2;;
    --height) HEIGHT="$2"; shift 2;;
    --max-disp) MAXDISP="$2"; shift 2;;
    --tile)  TILE="$2"; shift 2;;
    --gpu)   GPU="$2"; shift 2;;
    *)       OUT="$1"; shift;;   # 第一位置参 = outdir
  esac
done

SC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="${SC_WEIGHTS:-$SC/weights}"
mkdir -p "$OUT"
cd "$SC"

# ---- Step 0: 切片 + 自动转正 + scale 到 stage1 输入分辨率 ----
SS=""; [ -n "$START" ] && SS="-ss $START"
TO=""; [ -n "$END" ] && TO="-to $((END-START))"
ffmpeg -y -v error $SS $TO -i "$INPUT" \
  -vf "scale=${WIDTH}:${HEIGHT}" -c:v libx264 -crf 18 -an $OUT/_prep.mp4
echo "[0] prep done: $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $OUT/_prep.mp4)"

# ---- Step 1: DepthCrafter 深度 + forward-warp splatting (max_disp=平移量) ----
CUDA_VISIBLE_DEVICES=$GPU python depth_splatting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/DepthCrafter \
  --input_video_path $OUT/_prep.mp4 \
  --output_video_path $OUT/splatting_results.mp4 \
  --max_disp $MAXDISP \
  > $OUT/stage1.log 2>&1
echo "[1] depth+splatting done (max_disp=$MAXDISP)"

# ---- Step 2: 右眼 inpainting (tile_num=$TILE) ----
CUDA_VISIBLE_DEVICES=$GPU python inpainting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/StereoCrafter \
  --input_video_path $OUT/splatting_results.mp4 \
  --save_dir $OUT \
  --tile_num $TILE \
  >> $OUT/stage1.log 2>&1
echo "[2] right-eye inpainting done (tile=$TILE)"

# ---- Step 3: 左眼也过SVD(两眼协调) ----
# 左眼 = splatting 网格的 TL 象限; 构建网格 [左眼|黑; 白mask|左眼] → inpainting 重生成
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $OUT/_prep.mp4)
W2=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $OUT/_prep.mp4)
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $OUT/_prep.mp4)
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
echo "[3] left-eye SVD pass done"

# ---- Step 4: 拼 SBS = [SVD左 | SVD右] ----
RH=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $OUT/splatting_results_inpainting_results_sbs.mp4)
RW=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $OUT/splatting_results_inpainting_results_sbs.mp4)
HW=$((RW/2))
# 右半 = inpainting 生成的网格; 左眼pass的右半 = SVD左眼, 右眼inpainting的右半 = SVD右眼
ffmpeg -y -v error -i $OUT/splatting_results_inpainting_results_sbs.mp4 \
  -vf "crop=${HW}:${RH}:${HW}:0" $OUT/svd_right.mp4
ffmpeg -y -v error -i $OUT/left_pass_grid_inpainting_results_sbs.mp4 \
  -vf "crop=${HW}:${RH}:${HW}:0" $OUT/svd_left.mp4
for swap in "" "_SWAPPED"; do
  if [ -z "$swap" ]; then FC="[0:v][1:v]hstack=2[v]"; else FC="[1:v][0:v]hstack=2[v]"; fi
  ffmpeg -y -v error -i $OUT/svd_left.mp4 -i $OUT/svd_right.mp4 \
    -filter_complex "$FC" -map "[v]" -c:v libx264 -crf 16 -pix_fmt yuv420p \
    $OUT/final_sbs${swap}.mp4
done
echo "[4] final SBS: $OUT/final_sbs.mp4 (+_SWAPPED)"
echo "ALL DONE"
