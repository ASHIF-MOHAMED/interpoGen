# Real-Time Intermediate Flow Estimation for Video Frame Interpolation
## [YouTube](https://www.youtube.com/results?search_query=rife+interpolation&sp=CAM%253D) | [BiliBili](https://search.bilibili.com/all?keyword=SVFI&order=stow&duration=0&tids_1=0) | [Colab](https://colab.research.google.com/github/hzwer/ECCV2022-RIFE/blob/main/Colab_demo.ipynb) | [Tutorial](https://www.youtube.com/watch?v=gf_on-dbwyU&feature=emb_title) | [DeepWiki](https://deepwiki.com/hzwer/ECCV2022-RIFE)

## Introduction
This project is the implement of [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294). Currently, our model can run 30+FPS for 2X 720p interpolation on a 2080Ti GPU. It supports arbitrary-timestep interpolation between a pair of images.

**2024.08 - We find that [4.22.lite](https://github.com/hzwer/Practical-RIFE/tree/main?tab=readme-ov-file#model-list) is quite suitable for post-processing of [some diffusion model generated videos](https://drive.google.com/drive/folders/1hSzUn10Era3JCaVz0Z5Eg4wT9R6eJ9U9?usp=sharing).**

2023.11 - We recently release new [v4.7-4.10](https://github.com/hzwer/Practical-RIFE/tree/main#model-list) optimized for anime scenes! We draw from [SAFA](https://github.com/megvii-research/WACV2024-SAFA/tree/main)'s research.

2022.7.4 - Our paper is accepted by ECCV2022. Thanks to all relevant authors, contributors and users!

From 2020 to 2022, we submitted RIFE for five submissions（rejected by CVPR21 ICCV21 AAAI22 CVPR22). Thanks to all anonymous reviewers, your suggestions have helped to significantly improve the paper!

[ECCV Poster](https://drive.google.com/file/d/1xCXuLUCSwhN61kvIF8jxDvQiUGtLK0kN/view?usp=sharing) | [ECCV 5-min presentation](https://youtu.be/qdp-NYqWQpA) | [论文中文介绍](https://zhuanlan.zhihu.com/p/568553080) | [rebuttal (2WA1WR->3WA)](https://drive.google.com/file/d/16IVjwRpwbTuJbYyTn4PizKX8I257QxY-/view?usp=sharing) 

**Pinned Software: [RIFE-App](https://grisk.itch.io/rife-app) | [FlowFrames](https://nmkd.itch.io/flowframes) | [SVFI (中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation)**

16X interpolation results from two input images: 

![Demo](./demo/I2_slomo_clipped.gif)
![Demo](./demo/D2_slomo_clipped.gif)

## Quick Installation & Setup

We've made it easy to get started with interpoGen. Follow these simple steps:

### 1. Clone the Repository
```bash
git clone https://github.com/ASHIF-MOHAMED/interpoGen.git
cd interpoGen
```

### 2. Automatic Setup (Recommended)
Run our setup script to automatically install all dependencies and download model weights:

```bash
python setup.py
```

This script will:
- Install all required Python packages
- Download and set up FFmpeg
- Check for CUDA availability
- Download the model weights

### 3. Manual Setup (Alternative)
If you prefer to set up manually:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download model weights (if not already present)
# The model weights should be in train_log/flownet.pkl
```

You'll also need to install FFmpeg:
- **Windows**: Download from https://ffmpeg.org/download.html
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 4. GPU Acceleration (Optional)
For better performance, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Interpolate Images
```bash
python inference_img.py --img img0.png img1.png --exp=1
```
This will generate intermediate frames between img0.png and img1.png.

### Process a Video
```bash
python inference_video.py --video input.mp4 --exp=1 --output output.mp4
```

### Extract Frames from a Video/GIF
```bash
python runner1.py "path/to/video.mp4" extracted_frames
```

### Process Extracted Frames
```bash
python inference1_img.py
```
(Edit the input/output folders in the script as needed)

## Software
[Flowframes](https://nmkd.itch.io/flowframes) | [SVFI(中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) | [Autodesk Flame](https://vimeo.com/505942142) | [SVP](https://www.svp-team.com/wiki/RIFE_AI_interpolation) | [MPV_lazy](https://github.com/hooke007/MPV_lazy) | [enhancr](https://github.com/mafiosnik777/enhancr)

[RIFE-App(Paid)](https://grisk.itch.io/rife-app) | [Steam-VFI(Paid)](https://store.steampowered.com/app/1692080/SVFI/) 

We are not responsible for and participating in the development of above software. According to the open source license, we respect the commercial behavior of other developers.
