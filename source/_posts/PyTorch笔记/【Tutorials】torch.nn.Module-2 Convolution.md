---
title: 
date: 2019-09-22
tags:
categories: ["PyTorch笔记"]
mathjax: true
---


# 1d/2d/3d Convolution

看ppt吧，如果要总结，最好弄个动图吧

# nn.Conv2d

- in_channels: 输入通道数
- out_channels: 输出通道数，等价于卷积核个数
- kernel_size: 卷积核尺寸
- stride: 步长
- padding: 边界填充个数
- dilation: 空洞卷积大小. (空洞卷积常用语图像分割任务，主要作用是为了提升感受野，也就是说输出图像上的一个像素看到的是上一层图像更大的区域)
- groups: 分组卷积设置，表示分组卷积的组数。（一般用于模型轻量化，像ShuffleNet、SqueezeNet、MobileNet它们都有分组的概念，另外想AlexNet也是分组卷积，只不过AlexNet采用的分组卷积是由于硬件资源有限而采用的分两个GPU进行训练的。）
- bias: 偏置

**输出尺寸计算**：\
简化版：(不带padding, 也不带dilation)
$$out_{size} = \frac{in_{size} - kernel_{size}}{stride} + 1$$

完整版：
$$H_{out} = \frac{H_{in} + 2 \times padding[0] - dilation[0] \times (kernel_{size}[0] - 1) - 1}{stride[0]} + 1$$


**对老师这里讲得卷积核和input的计算方式表示怀疑**
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv1.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp;2维卷积3维Tensor(不考虑batch那一维)</center>


# Transpose Convolution
&emsp; 转置卷积(Transpose Convolution)又称为反卷积(Deconvolution)或者部分跨越卷积(Fractionally-strided Convolution), 用于对图像进行上采样(UpSample)，这在图像分割任务中会经常被使用。

> 为了避免与《信号与系统》中的反卷积混淆，一般还是叫转置卷积(Transpose Convolution)

为什么称为转置卷积?

正常卷积: \
假设图像尺寸为$4 \times 4$, 卷积核为 $3 \times 3$ , padding=0, stride=1 \
则在代码中会将图片和卷积核都转换为矩阵，转换后的矩阵形式如下：\
图像: $I_{16 \times 1}$ , 即图像会被展平\
卷积核: $K_{4 \times 16}$ , 下标4表示输出有4行（这需要根据input、kernel、padding、stride、dilation这些值来计算），下标16是先将卷积核展平($3 \times 3 = 9$)再填充0至16个数的长度。\
输出: $O_{4 \times 1} = K_{4 \times 16} \cdot I_{16 \times 1}$

转置卷积: \
假设图像尺寸为 $2 \times 2$ , 卷积核为 $3 \times 3$ , padding=0, stride=1 \
则在代码中会将图片和卷积核都转换为矩阵，转换后的矩阵形式如下：\
图像: $I_{4 \times 1}$ , 即图像会被展平 \
卷积核: $K_{16 \times 4}$ \
输出: $O_{16 \times 1} = K_{16 \times 4} \cdot I_{4 \times 1}$



# 参考文献
[1] DeepShare.net > PyTorch框架
