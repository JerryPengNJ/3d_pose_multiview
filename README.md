# 3d_pose_multiview

## This is the offical implementation of the paper "A Simple baseline for multi-view 3d human pose estimation " in PyTorch.


## Dataset

We use the Human3.6M dataset for training and evaluation. The dataset can be downloaded from the [Human3.6M website](http://vision.imar.ro/human3.6m/description.php). The dataset should be placed in the `data` folder.

We process the data follow the [videopose3d](https://github.com/facebookresearch/VideoPose3D) repository.


## Training

``` 
python main.py
```