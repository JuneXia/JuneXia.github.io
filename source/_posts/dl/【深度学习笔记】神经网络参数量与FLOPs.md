---
title: 【深度学习笔记】神经网络参数量与FLOPs
date: 2019-03-02 17:28:05
tags:
categories: ["深度学习笔记"]
mathjax: true
---

# 一些基本概念
FLOPS：注意S大写，是 floating point operations per second 的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。
<!-- more -->

FLOPs：注意s小写，是 floating point operations 的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。

本文讨论的是算法模型，应指的是FLOPs。

以下答案不考虑 activation function 的运算。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/conv2d_6.jpg" width = 100% height = 100% />
</div>

# 对于卷积层来说
假设有：\
inputs: [$batch\_size, I_{H}, I_{W}, C_{in}$] = [b, 28, 28, 3] \
kernel: [$C_{out}, K_{H}, K_{W}, C_{in}$] = [16, 5, 5, 3] \
bias: [$C_{out}$] = [16]

## 参数量
&emsp; 这里约定，一个kernel有$C_{out}$个卷积核，或者叫有$C_{out}$个output channel.

一个卷积核的参数量：$K_{H} \times K_{W}$；\
一个output channel的参数量：因为有 $C_{in}$ 个 input channel，则一个 output channel 有 $K_{H} \times K_{W} \times C_{in}$ 个参数；\
$C_{out}$ 个 output channel 的参数量：$C_{out} \times K_{H} \times K_{W} \times C_{in}$ \
加上bias，这一层总的参数量为 $C_{out} \times K_{H} \times K_{W} \times C_{in} + C_{out}$


## 计算量
一个卷积核和一个input map做一次乘法的计算量：$K_{H} \times K_{W}$，$C_{in}$ 个卷积核和 $C_{in}$ 个input map 做一次乘积的计算量：$K_{H} \times K_{W} \times C_{in}$. \
做完乘法还要做累加，$C_{in}$个卷积核和$C_{in}$ 个input map 做一次内积的计算量：
$$K_{H} \times K_{W} \times C_{in} + K_{H} \times K_{W} \times C_{in} - 1  \tag{1}$$
（n个数相加的加法次数为n-1），若考虑bias，则是：
$$K_{H} \times K_{W} \times C_{in} + K_{H} \times K_{W} \times C_{in} - 1 + 1 \tag{2}$$
式(1)、(2)也可以称作是一个output channel的计算量。

那么 $C_{out}$ 个 output channel 的计算量就是：
$$(K_{H} \times K_{W} \times C_{in} + K_{H} \times K_{W} \times C_{in} - 1) \times C_{out}  \tag{3}$$

以上卷积核才计算了一次，对于一个 $I_{H} \times I_{W}$ 的 input map，假设输出不改变原尺寸大小，则这个 kernel 在input map上的总计算量为：
$$(K_{H} \times K_{W} \times C_{in} + K_{H} \times K_{W} \times C_{in} - 1) \times C_{out} \times I_{H} \times I_{W} \tag{4}$$


# 对全连接层来说
假设有：\
inputs: [$batch\_size, N_{in}$] = [b, 128] \
weight: [$N_{out}, N_{in}$] = [64, 128] \
bias: [$N_{out}$] = [64]

## 参数量
$N_{out} \times N_{in} + N_{out}$

## 计算量
不考虑bias:
$(N_{in} + N_{in} - 1) \times N_{out}$

考虑bias:
$(N_{in} + N_{in}) \times N_{out}$


---
&emsp; 最后还要说一点关于FLOPs的计算，在知乎上也有[讨论](https://www.zhihu.com/question/65305385/answer/256845252)，另外Nvidia的Pavlo Molchanov等人的[文章](https://arxiv.org/pdf/1611.06440.pdf)的APPENDIX中也有介绍，由于是否考虑biases，以及是否一个MAC算两个operations等因素，最终的数字上也存在一些差异。但总的来说，计算FLOPs其实也是在对比之下才显示出某种算法，或者说网络的优势，如果我们坚持一种计算标准下的对比，那么就是可以参考的，有意义的计算结果。

