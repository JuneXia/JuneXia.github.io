---
title: 【开发环境与工具】conda安装paddle
date: 2020-2-09
tags:
categories: ["开发环境与工具"]
mathjax: true
---

&emsp; 我按官方文档 [使用conda安装paddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/install_Conda.html)
步骤安装失败，好多坑啊。这里结合官方文档踩坑记录如下：
<!-- more -->

step1: 安装conda

已经安装过了，这里就不讲了

step2: 创建conda虚拟环境
```bash
$ conda create -n paddle_env python=3.5
```

step3: 激活虚拟环境
```bash
$ source activate paddle_env
# 或者
$ conda activate paddle_env

$ which python
/disk1/software/anaconda3/envs/paddle_env/bin/python
```

step4: 在conda虚拟环境下，使用pip安装各种python库
```bash
$ pip -V  # 查看是否确实是conda 虚拟环境下的pip
pip 10.0.1 from /disk1/software/anaconda3/envs/paddle_env/lib/python3.5/site-packages/pip (python 3.5)

# 开始使用pip安装各种所需要的python库
$ pip install numpy
$ pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
$ pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
$ pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```


