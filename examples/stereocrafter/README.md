# StereoCrafter 使用方案（定档版）

基于 [StereoCrafter](https://github.com/TIGER-AI-Lab/StereoCrafter)（DepthCrafter 深度 + Forward-Warp 渲染 + SVD inpainting 修复空洞），在 RTX 3090 24GB 上实测敲定的 2D→3D 方案。

> **用户判定: SC 04(1080p+tile)为选型**。03 无tile的新生成清晰度不够; 04 tile 接缝不明显。并建议**两眼协调化**(见关键结论6)。

## 定档规格

| 项 | 定档值 | 说明 |
|---|---|---|
| 流程 | **Stage1**: DepthCrafter 深度 + forward-warp splatting → **Stage2**: SVD inpainting 右眼 → **Stage3**: 左眼也过 SVD(两眼协调) → **Stage4**: 拼 SBS=[SVD左\|SVD右] | 见 run_sc_d_portrait.sh 完整流程 |
| 输入分辨率 | **1080p + tile_num=2**(04 定档) | 1024×576 无tile(03)只作干净基准对比 |
| 深度 | DepthCrafter, max_res=1024(默认), window_size=70 | 输入 >1024 自动降采样 |
| 视差 | `--max_disp 20` | 实际块视差 -10~0px |
| 输出 | 最终 `final_sbs.mp4`(3840×1024, 25fps, 原帧数) + SWAPPED | 两眼均过 SVD, 色彩/清晰度一致 |
| 平移量 | `--max-disp <px>`(默认 20) | 实际块视差峰值~-10px, 线性 |

## 关键结论（实测）

1. **24GB 显存上限 → 输入分辨率是关键**:
   - **1080p(1920×1080)无 tile 会 OOM**: splatting 输出 2×2 网格(3840×2160), inpainting quadrant 1920×1080→1920×1024, VAE 编码在无 tile 全帧下爆 24GB(4.69GB 请求失败)。
   - **1024×576 输入**(SVD 原生尺寸)无 tile 可行: quadrant 1024×576, inpainting 全帧约 16GB。
   - **4K 输入即使 tile_num=2 也 OOM**: 4K quadrant 3840×2048, 2×2 tile 每个 1984×1088(latent 248×136=33728 tokens)≈ 03 无 tile 的 1920×1024(30720 tokens)一样大。
   - **04 组改 1080p + tile_num=2**: 每 tile 1024×576(SVD 原生), 可跑。**接缝不明显(用户确认)**。
2. **`expandable_segments` 不可用**: SC env 是 torch 2.0.1, 不支持该 PYTORCH_CUDA_ALLOC_CONF 选项(2.1+ 才有), 直接报 `Unrecognized CachingAllocator option`。
3. **CLI 用 fire.Fire**: `depth_splatting_inference.py` 的 `main(input_video_path, output_video_path, unet_path, pre_trained_path, max_disp, process_length, batch_size)` / `inpainting_inference.py` 的 `main(pre_trained_path, unet_path, input_video_path, save_dir, frames_chunk, overlap, tile_num)`。参数名直接对应。
4. **权重来源**: SVD(stabilityai/stable-video-diffusion-img2vid-xt-1-1)**HF gated 403 → 走 ModelScope**(~10GB); DepthCrafter + StereoCrafter unet 走 hf-mirror。Forward-Warp CUDA 扩展需编译(CUDA11.1 nvcc 编 torch11.8, major 相同仅 warn)。
5. **03 新生成清晰度不够(用户确认)**: 右眼经 SVD inpainting 偏软, 原图左眼过锐 → 两眼不协调。
6. **两眼协调化(用户建议, 已实现)**: 右眼过 SVD, 左眼是原图 → 清晰度/色彩不匹配。**把左眼也过一遍 SVD**(构建 TL=左眼/BL=全白mask/BR=左眼 的网格跑 inpainting), 两眼都经过 SVD 生成。实测结果: 清晰度左19.4/右15.8(接近), 饱和度77.6/76.0, 色彩 R105/G137/B141 vs R105/G138/B142(一致), 视差 std 3.94 保留。
7. **竖屏素材坑**: 手机竖拍带 rotation=90 → 用 `ffmpeg -vf scale=1080:1920`(默认 autorotate), 不要手动 transpose(画面倾倒)。

## 目录结构

```
examples/stereocrafter/
├── README.md                    本文件
└── scripts/
    ├── run_sc_horizontal.sh     生产脚本(横屏): 切片+scale1080p+SC04+左眼SVD+拼SBS,
    │                             支持平移量(--max-disp) / tile / GPU / outdir
    └── run_sc_d_portrait.sh     竖屏完整流程(autorotate+SC04+左眼SVD+拼SBS)
                                支持 --max-disp / --tile / --gpu / --out / 切片
```

> 注: StereoCrafter 无需改源码, 所有调整都是输入分辨率/tile 参数层面的运行配置。Forward-Warp 扩展编译在 `pip install -e .` 时完成。

## 环境安装

```bash
# conda env (py3.8) + torch 2.0.1 cu118 + xformers 0.0.20 + requirements
conda create -n sc python=3.8 -y
pip install torch==2.0.1+cu118 torchvision==0.15.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# Forward-Warp CUDA 扩展(需 nvcc 在 PATH)
pip install -e ./dependency/Forward_Warp   # 实测: nvcc 11.1 编 torch 11.8, major相同仅warn

# 权重放到 <仓库>/weights/ (脚本默认找这里; 或 export SC_WEIGHTS=<其他路径>)
# SVD(stabilityai/stable-video-diffusion-img2vid-xt-1-1): HF gated 403 → 用 ModelScope 下(~10GB)
# DepthCrafter: hf-mirror → weights/DepthCrafter
# StereoCrafter unet: hf-mirror → weights/StereoCrafter
```

## 运行（推荐: 生产脚本）

```bash
# 前置: 在装有 StereoCrafter 依赖的 env 里运行(python 走 PATH); 权重在 <仓库>/weights/

# 横屏全流程(切片+scale1080p+深度+splatting+右眼SVD+左眼SVD+拼SBS+SWAPPED)
./scripts/run_sc_horizontal.sh input.mp4 out/

# 加平移量(视差强度): --max-disp 40; 切片 30~45s; GPU1
./scripts/run_sc_horizontal.sh input.mp4 out/ \
  --max-disp 40 --start 30 --end 45 --gpu 1

# 竖屏素材(自动 autorotate)
./scripts/run_sc_d_portrait.sh input.mp4              # 全片
./scripts/run_sc_d_portrait.sh input.mp4 30 45 --max-disp 40
./scripts/run_sc_d_portrait.sh input.mp4 --out out/   # 指定输出目录(默认 ./sc_output)
```

## 运行（定档流程手动步骤: 04 + 左眼SVD + 拼SBS）

```bash
# Step1: 深度 + splatting (1080p 输入)
CUDA_VISIBLE_DEVICES=0 python depth_splatting_inference.py \
  --pre_trained_path weights/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path weights/DepthCrafter \
  --input_video_path input.mp4 \
  --output_video_path out/splatting_results.mp4 --max_disp 20

# Step2: 右眼 inpainting (tile_num=2)
CUDA_VISIBLE_DEVICES=0 python inpainting_inference.py \
  --pre_trained_path weights/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path weights/StereoCrafter \
  --input_video_path out/splatting_results.mp4 --save_dir out --tile_num 2

# Step3: 左眼也过SVD(两眼协调) → 构建左眼pass网格
ffmpeg -i out/splatting_results.mp4 -vf "crop=1920:1080:0:0" out/left_eye.mp4
ffmpeg -y -i out/left_eye.mp4 -f lavfi -i color=black:s=1920x1080:r=25:d=15.5 \
  -f lavfi -i color=white:s=1920x1080:r=25:d=15.5 -i out/left_eye.mp4 \
  -filter_complex "[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]" \
  -map "[v]" out/left_pass_grid.mp4
CUDA_VISIBLE_DEVICES=0 python inpainting_inference.py \
  --pre_trained_path weights/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path weights/StereoCrafter \
  --input_video_path out/left_pass_grid.mp4 --save_dir out --tile_num 2

# Step4: 拼 SBS = [SVD左 | SVD右] (各自取 inpainting 输出SBS的右半)
ffmpeg -i out/splatting_results_inpainting_results_sbs.mp4 -vf "crop=1920:1024:1920:0" out/svd_right.mp4
ffmpeg -i out/left_pass_grid_inpainting_results_sbs.mp4 -vf "crop=1920:1024:1920:0" out/svd_left.mp4
ffmpeg -y -i out/svd_left.mp4 -i out/svd_right.mp4 -filter_complex "[0:v][1:v]hstack=2[v]" \
  -map "[v]" out/final_sbs.mp4

# SWAPPED(用户交叉眼观看)
ffmpeg -y -i out/svd_left.mp4 -i out/svd_right.mp4 -filter_complex "[1:v][0:v]hstack=2[v]" \
  -map "[v]" out/final_sbs_SWAPPED.mp4

# 竖屏素材: 直接用 run_sc_d_portrait.sh (含 autorotate + 全流程)
./scripts/run_sc_d_portrait.sh input.mp4
```
