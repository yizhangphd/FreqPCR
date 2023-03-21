<!-- # FreqPCR -->
# Frequency-Modulated Point Cloud Rendering with Easy Editing

This repository contains the official implementation for the paper: **[Frequency-Modulated Point Cloud Rendering with Easy Editing](https://arxiv.org/abs/2303.07596) (CVPR 2023 Highlight)**.

<img src="image/hotdog_chair.gif" width = "150" height = "150" alt="chair" /><img src="image/lego_chair.gif" width = "150" height = "150" alt="chair" /><img src="image/family_ficus.gif" width = "150" height = "150" alt="chair" /><img src="image/materials_drums.gif" width = "150" height = "150" alt="chair" />
 
Full code, configs and data will be available later.

## Installation

```
conda create -n FreqPCR python=3.8
conda activate FreqPCR

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install matplotlib
pip install opencv-python
pip install lpips
pip install git+https://github.com/francois-rozet/piqa
pip install tensorboard
pip install ConfigArgParse
pip install open3d

# PyTorch3D rasterization
python setup.py develop
```
<!-- We provide two ways for point cloud rasterization.
**For headless servers**, we recommend running the following command to install the rasterization module provided by [PyTorch3D](https://github.com/facebookresearch/pytorch3d): -->

**Optionally**, for real-time rendering, please run the following command:
```
# OpenGL, borrowed from NPBG
pip install \
    Cython \
    PyOpenGL \
    PyOpenGL_accelerate

# need to install separately
pip install \
    git+https://github.com/DmitryUlyanov/glumpy \
    numpy-quaternion

# pycuda
git clone https://github.com/inducer/pycuda
cd pycuda
git submodule update --init
export PATH=$PATH:/usr/local/cuda/bin
./configure.py --cuda-enable-gl
python setup.py install
cd ..
```

## Data Preparation

The layout should look like this

```
FreqPCR
├── data
    ├── nerf_synthetic
    ├── TanksAndTemple
    ├── dtu
    |   ├── dtu_110
    │   │   │── cams_1
    │   │   │── image
    │   │   │── mask
    │   │   │── pc.ply
    |   ├── dtu_114
    |   ├── dtu_118
    ├── scannet
    │   │   │──0000
    |   │   │   │──color_select
    |   │   │   │──pose_select
    |   │   │   |──intrinsic
    |   │   │   |──pc.ply
    │   │   │──0043
    │   │   │──0045
```

- **NeRF-Synthetic**: Please download the dataset provided by [NeRF](https://github.com/bmild/nerf) and put the unpacked files in ``data/nerf_synthetic``. 
Since [Point-NeRF](https://github.com/Xharlie/pointnerf) provide an implementation of point cloud generation using [MVSNet](https://github.com/YoYo000/MVSNet), we can run [Point-NeRF](https://github.com/Xharlie/pointnerf) and save the point clouds in ``data/pc/``.
You can also download raw point clouds from [here](https://drive.google.com/drive/folders/1qcEk97RgwCAzzmXUTUXCGNsGzwYPicLA).

- **Tanks and Temples**: Please download the dataset provided by [NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip) and put the unpacked files in ``data/TanksAndTemple``. To generate point clouds, run [Point-NeRF](https://github.com/Xharlie/pointnerf) and save the point clouds in ``data/pc/``.

- **DTU**: Please download images and masks from [IDR](https://github.com/lioryariv/idr) and camera parameters from [PatchmatchNet](https://github.com/FangjinhuaWang/PatchmatchNet). We use the point clouds provided by [NPBG++](https://github.com/rakhimovv/npbgpp).

- **ScanNet**: Please download data from [ScanNet](http://www.scan-net.org/) and run ``select_scan.py`` to select the frames. We use the provided depth map to generate point clouds.
For ``scene0043_00``, the frames after 1000 are ignored because the camera parameters are ``-inf``.

## Running

We experiment on a single NVIDIA GeForce RTX 3090 (24G), if your GPU memory is not enough, you can set the ``batch_size`` to 1 or reduce the ``train_size``.

#### Point Cloud Geometry Optmization
```
python pc_opt.py --config=configs/pc_opt/nerf_hotdog.txt
```
The optimized point cloud will be saved in ``logs_pc_opt/``

#### Rasterization
```
python run_rasterize.py --config=configs/render/nerf_hotdog.txt
```
<!-- This is PyTorch3D rasterization for headless computers.  -->
The point index and depth buffers will be saved in ``data/fragments``.

#### Training

```
python train.py --config=configs/render/nerf_hotdog.txt
```
The results will be saved in ``logs/``. You can also run tensorboard to monitor training and testing.

#### Real-time Rendering
```
python inference_gl.py --config=configs/render/nerf_hotdog.txt
```

#### Editing
```
TODO
```


## Acknowledgements
- This project builds on our previous work [RadianceMapping](https://github.com/seanywang0408/RadianceMapping).
- The code of rasterization borrows a lot from [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and [NPBG](https://github.com/alievk/npbg).

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhang2023frequency,
  title={Frequency-Modulated Point Cloud Rendering with Easy Editing},
  author={Zhang, Yi and Huang, Xiaoyang and Ni, Bingbing and Li, Teng and Zhang, Wenjun},
  journal={arXiv preprint arXiv:2303.07596},
  year={2023}
}
```
