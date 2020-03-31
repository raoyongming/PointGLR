PointGLR
===
This repository contains the PyTorch implementation for paper __Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds__ (CVPR 2020) \[[arXiv](https://arxiv.org/abs/2003.12971)\]

![overview](https://raoyongming.github.io/files/fig_PointGLR.jpg)

If you find our work useful in your research, please consider citing:
```
@inproceedings{rao2020global,
  title={Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds},
  author={Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```


## Usage

### Requirement

- Python 3
- Pytorch 0.4
- CMake > 2.8

**Note**: The code is not not compatible with Pytorch >= 1.0 due to the C++/CUDA extensions. 

### Building C++/CUDA Extensions for PointNet++

```
mkdir build && cd build
cmake .. && make
```

### Dataset Preparation

- Download ModelNet point clouds (XYZ and normal):
```
mkdir dataset && cd dataset
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
```
- Preprocess dataset:
```
CUDA_VISIBLE_DEVICES=0 python tools/prepare_modelnet.py
```

### Training & Evaluation
To train an SSG PointNet++ model on ModelNet:
```
bash train.sh exp_name pointnet2 modelnet
```
To train an SSG RSCNN model on ModelNet:
```
bash train.sh exp_name pointnet2 modelnet
```
You can  modify `multiplier` in `cfgs/config.yaml` to train larger models. As a reference, the unsupervisedly trained 1x SSG PointNet++ and 1x SSG RSCNN models should have around 92.2% accuracy on ModelNet40. By increasing channel width (4x~5x), our best PointNet++ and RSCNN models achieved around 93.0% accuracy. The results might vary by 0.2%~0.5% between identical runs due to different random seed.

## Acknowledgement

The code is based on [Relation-Shape CNN](https://github.com/Yochengliu/Relation-Shape-CNN) and [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Contact
If you have any questions about our work, please contact <raoyongming95@gmail.com>
