---
title: 【开发环境与工具】ubuntu python导入自己的package
date: 2019-05-13
tags:
categories: ["开发环境与工具"]
mathjax: true
---

**问题背景**：随着代码的积累，发现很多自己写的代码是可以通用的，可以将自己的代码库加入系统路径，方便其他python代码在任何地方都能够使用。
<!-- more -->

假设我的项目目录结构如下：
```
xxx@ailab-server:~/xxx/project$ tree
.
├── cifar
│   └── main.py
├── example.py
├── libml
│   ├── datasets
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── metrics
│   │   ├── evaluate.py
│   │   ├── evaluate_test_roc.py
│   │   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── lenet.py
│   │   └── resnet.py
│   └── utils
│       ├── config.py
│       ├── __init__.py
│       └── tools.py
└── tutorial
    ├── evaluate.py
    ├── tensor.py
```

我希望将 project下的 libml 目录设置为我的公用 package，解决方案有如下：


> 注意：实测下面的解决方案只有在3在终端和pycharm远程连接中都管用，而解决方案1和2只能在终端命令行中管用，所以推荐使用方案3。



**解决方案1**
```bash
$ vi ~/.bashrc

# 文件末尾加入如下代码
export PYTHONPATH=$PYTHONPATH:/path to your/project

$ source ~/.bashrc
```
这种方法只能是当前用户，且在/usr/bin/python解释器下可用，在其他用户以及conda环境下都不可用。

> 注意：导入路径不是 /path to your/project/libml，而是 /path to your/project



**解决方案2**
```bash
$ sudo vi /etc/profile

# 文件末尾加入如下代码
export PYTHONPATH=$PYTHONPATH:/path to your/project

$ source /etc/profile
```
这种方法对所有用户都可用，但也只能在/usr/bin/python解释器下可用，在conda环境下都不可用。


**解决方案3**
```bash
# 检查自己的 python3 路径在哪里
$ which python3
/path to your/anaconda3/envs/paddle_env/bin/python3

# cd 该python环境的packages目录
$ cd /path to your/anaconda3/envs/paddle_env/lib/python3.5/site-packages

# 新建一个 pth 文件
$ sudo touch mypackage.pth

# 填入自己的 package 路径
/path to your/project
```

