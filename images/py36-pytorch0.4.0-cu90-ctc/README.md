# py36-pytorch0.4.0-cu90-ctc

## Feature

* CUDA 9.0 
* CUDNN 7.1.2
* python 3.6
* pytorch 0.4.0
* tensorboardx
* ctcdecode
* warpctc-pytorch
* ffmpeg 4.0.2

+ ipython
+ lmdb
+ h5py

## Build

```bash
docker build -t py36-pytorch0.4.0-cu90-ctc ./images/py36-pytorch0.4.0-cu90-ctc/
```

## Note
最后一步编译会出现processing ctcdecode，后面比较慢，请耐心等待15分钟左右，谢谢