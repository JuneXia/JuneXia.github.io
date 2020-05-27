---
title: 
date: 2020-05-08
tags:
categories: ["PyTorch笔记"]
mathjax: true
---


get pytorch source code
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

```bash
python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

主要更改成下面几项，其他选项默认。（当然这具体得根据您的需求）
```
AT_INSTALL_SHARE_DIR             share         # 使用share表示编译成动态库，否则表示编译成静态库
BUILD_CAFFE2_MOBILE              OFF
BUILD_CAFFE2_OPS                 OFF
BUILD_TEST                       OFF
CAFFE2_STATIC_LINK_CUDA          OFF           # 如果不使用cuda则这里一并去掉
CMAKE_INSTALL_PREFIX             /home/xiajun/program/temp/pytorch/torch_static    # 指定您的安装目录
INSTALL_GTEST                    OFF           # test可以不用安装
USE_C10D_GLOO                    OFF           # 如果不使用cuda则这里一并去掉
USE_C10D_MPI                     OFF           # 如果不使用cuda则这里一并去掉
USE_C10D_NCCL                    OFF           # 如果不使用cuda则这里一并去掉
USE_CUDA                         OFF           # 选择是否使用cuda                                  
```

```bash
python setup.py install  # 可能要给sudo权限
```

异常处理

**异常1**：找不到 fbgemm 等第三方库问题

解决办法：\
可事先从 github 将 fbgemm 下载好，然后将其放到 /project of pytorch/third_party/fbgemm 目录下。


**异常2**：
```bash
/usr/bin/ld: can not find: -lshm
/usr/bin/ld: can not find: -ltorch_python
```

解决办法：\
先查看libshm、libtorch_python 在什么位置：
```bash
$ locate libshm
/usr/local/lib/python3.5/dist-packages/torch/lib/libshm.so

$ locate libtorch_python
/usr/local/lib/python3.5/dist-packages/torch/lib/libtorch_python.so
```

然后建立软连接
```bash
$ cd /project of pytorch/torch
$ ln -s /usr/local/lib/python3.5/dist-packages/torch/lib/libshm.so ./
$ ln -s /usr/local/lib/python3.5/dist-packages/torch/lib/libtorch_python.so ./
```

# 参考文献
[1] [pytorch github readme](https://github.com/pytorch/pytorch)
[2] [/usr/bin/ld 找不到 -lshm -ltorch_python等问题](https://github.com/pytorch/pytorch/issues/23554)

