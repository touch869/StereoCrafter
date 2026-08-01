#!/bin/bash
# StereoCrafter 竖屏素材(1080x1920 portrait)完整流程
# ==================================================
# 依据 SC 04(1080p+tile_num=2)定档 + 两眼协调化(左眼也过SVD)方案。
# 竖屏坑: 手机竖拍视频带 rotation=90 元数据, 必须用 ffmpeg scale 默认 autorotate 转正,
#         不要手动 transpose(会在错误基准上转90°→画面倾倒)。
#
# 用法:
#   ./run_sc_d_portrait.sh /path/to/input.mp4                      # 全片
#   ./run_sc_d_portrait.sh /path/to/input.mp4 30 45                # 只处理 [30,45) 秒片段
#   ./run_sc_d_portrait.sh /path/to/input.mp4 30 45 --max-disp 40  # 加平移量(视差强度)
#   ./run_sc_d_portrait.sh /path/to/input.mp4 --gpu 1 --tile 2
#   ./run_sc_d_portrait.sh /path/to/input.mp4 --out /path/to/outdir  # 指定输出目录
# 选项:
#   --out <dir>       输出目录(默认 ./sc_output, 相对当前目录; 不硬编码绝对路径)
#   --max-disp <px>   平移量(默认 20, 线性)
#   --tile <N>        默认 2
#   --gpu <N>         默认 0
# 输出: {out}/final_sbs.mp4 (SVD左|SVD右) + final_sbs_SWAPPED.mp4
#
# 可移植性: 仓库根由脚本位置推导; 权重目录用 SC_WEIGHTS 环境变量(默认 <仓库>/weights);
#           在装有 SC 依赖的 env 里运行(python 走 PATH)。
# ==================================================

set -e
IN_VIDEO="${1:?用法: $0 <视频路径> [起始秒] [结束秒] [选项]}"
shift
START=0; END=""; MAXDISP=20; TILE=2; GPU=0; OUT=./sc_output; POS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-disp) MAXDISP="$2"; shift 2;;
    --tile)     TILE="$2"; shift 2;;
    --gpu)      GPU="$2"; shift 2;;
    --out)      OUT="$2"; shift 2;;
    *)
      if [ $POS -eq 0 ]; then START="$1"; POS=1
      elif [ $POS -eq 1 ]; then END="$1"; POS=2; fi
      shift;;
  esac
done
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
W="${SC_WEIGHTS:-$SC/weights}"
mkdir -p "$OUT"
cd "$SC"

BASE=$(basename "$IN_VIDEO" .mp4)

# ---- Step 0: autorotate 转正竖屏 1080x1920 ----
# scale 默认应用 rotation 元数据 → 正确竖屏; 切片用 -ss/-to 放 -i 前(快速定位)
SS=""; [ -n "$START" ] && SS="-ss $START"
TO=""; [ -n "$END" ] && TO="-to $((END-START))"
ffmpeg -y -v error $SS $TO -i "$IN_VIDEO" \
  -vf "scale=1080:1920" -c:v libx264 -crf 18 -an $OUT/${BASE}_auto.mp4
echo "[0] autorotate done: $(ffprobe -v error -show_entries stream=width,height -of csv=p=0 $OUT/${BASE}_auto.mp4)"

# ---- Step 1: DepthCrafter 深度 + forward-warp splatting ----
CUDA_VISIBLE_DEVICES=$GPU python depth_splatting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/DepthCrafter \
  --input_video_path $OUT/${BASE}_auto.mp4 \
  --output_video_path $OUT/splatting_results.mp4 \
  --max_disp $MAXDISP \
  > $OUT/stage1.log 2>&1
echo "[1] depth+splatting done (max_disp=$MAXDISP)"

# ---- Step 2: 右眼 inpainting (tile_num=2, 竖屏quadrant 1024x1920) ----
CUDA_VISIBLE_DEVICES=$GPU python inpainting_inference.py \
  --pre_trained_path $W/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path $W/StereoCrafter \
  --input_video_path $OUT/splatting_results.mp4 \
  --save_dir $OUT \
  --tile_num $TILE \
  >> $OUT/stage1.log 2>&1
echo "[2] right-eye inpainting done (tile=$TILE)"

# ---- Step 3: 左眼也过SVD(两眼协调, 见 README) ----
# 构建左眼pass网格: TL=左眼, TR=黑, BL=全白mask, BR=左眼 → 整帧重新生成
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $OUT/${BASE}_auto.mp4)
W2=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $OUT/${BASE}_auto.mp4)
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $OUT/${BASE}_auto.mp4)
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
# SVD右 = 右眼inpainting输出SBS的右半; SVD左 = 左眼pass输出SBS的右半
RH=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $OUT/splatting_results_inpainting_results_sbs.mp4)
RW=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $OUT/splatting_results_inpainting_results_sbs.mp4)
HW=$((RW/2))
# SVD右眼 = SC04右眼SBS右半; SVD左眼 = 左眼pass SBS右半
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
