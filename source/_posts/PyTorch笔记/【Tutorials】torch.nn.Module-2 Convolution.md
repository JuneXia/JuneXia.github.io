---
title: 
date: 2019-09-22
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述卷积、转置卷积以及它们的 PyTorch nn.Module 中的实现方法。
<!-- more -->

# 1d/2d/3d Convolution

**卷积运算**: 卷积核在输入信号(图像)上滑动, 相应位置上进行**乘加** \
**卷积核**: 又称为滤波器, 过滤器, 可认为是某种模式, 某种特征。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv01.jpg" width = 60% height = 60% />
</div>

卷积过程类似于用一个模版去图像上寻找与它相似的区域, 与卷积核模式越相似, 激活值越高, 从而实现特征提取。
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv02.jpg" width = 60% height = 60% />
</div>

AlexNet卷积核可视化,发现卷积核学习到的是**边缘, 条纹, 色彩**这一些细节模式.
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv03.jpg" width = 60% height = 60% />
</div>


## nn.Conv2d
```python
class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)
```
- **in_channels**: 输入通道数
- **out_channels**: 输出通道数，等价于卷积核个数
- **kernel_size**: 卷积核尺寸
- **stride**: 步长
- **padding**: 边界填充个数
- **dilation**: 空洞卷积大小. (空洞卷积常用语图像分割任务，主要作用是为了提升感受野，也就是说输出图像上的一个像素看到的是上一层图像更大的区域)
- **groups**: 分组卷积设置，表示分组卷积的组数。（一般用于模型轻量化，像ShuffleNet、SqueezeNet、MobileNet它们都有分组的概念，另外想AlexNet也是分组卷积，只不过AlexNet采用的分组卷积是由于硬件资源有限而采用的分两个GPU进行训练的。）
- **bias**: 偏置

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


代码示例：
```python
# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_convolution.py
# @author     : tingsongyu
# @date       : 2019-09-23 10:08:00
# @brief      : 学习卷积层
"""
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from tools.common_tools import transform_invert, set_seed

set_seed(3)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ 2d
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)

# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
```

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv4.jpg" width = 60% height = 60% />
</div>
<center>图2 &nbsp;nn.Conv2d卷积可视化与原图比较(左侧为原图，右侧为卷积后的效果图)</center>


# Transpose Convolution
&emsp; 转置卷积(Transpose Convolution)又称为反卷积(Deconvolution)或者部分跨越卷积(Fractionally-strided Convolution), 用于对图像进行上采样(UpSample)，这在图像分割任务中会经常被使用。

> 为了避免与《信号与系统》中的反卷积混淆，一般还是叫转置卷积(Transpose Convolution)


<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv3.jpg" width = 50% height = 50% />
                </div>
            </td>
            <center>(a) &nbsp; 正常卷积(下采样)</center>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv2.jpg" width = 50% height = 50% />
                </div>
            </td>
            <center>(b) &nbsp; 转置卷积(上采样)</center>
        </tr>
    </table>
    <center>图3 &nbsp; 正常卷积与转置卷积</center>
</html>


为什么称为转置卷积?

**正常卷积**: \
假设图像尺寸为$4 \times 4$, 卷积核为 $3 \times 3$ , padding=0, stride=1 \
则在代码中会将图片和卷积核都转换为矩阵，转换后的矩阵形式如下：\
图像: $I_{16 \times 1}$ , 即图像会被展平\
卷积核: $K_{4 \times 16}$ , 下标4表示输出有4行（这需要根据input、kernel、padding、stride、dilation这些值来计算），下标16是先将卷积核展平($3 \times 3 = 9$)再填充0至16个数的长度，下面表格给出了这一操作的示意图。\
输出: $O_{4 \times 1} = K_{4 \times 16} \cdot I_{16 \times 1}$

| Input | $I_{1}$ | $I_{2}$ | $I_{3}$ | $I_{4}$ | $I_{5}$ | $I_{6}$ | $I_{7}$ | $I_{8}$ | $I_{9}$ | $I_{10}$ | $I_{11}$ | $I_{12}$ | $I_{13}$ | $I_{14}$ | $I_{15}$ | $I_{16}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $Kernel$ | $K_{1}$ | $K_{2}$ | $K_{3}$ | $0$ | $K_{4}$ | $K_{5}$ | $K_{6}$ | $0$ | $K_{7}$ | $K_{8}$ | $K_{9}$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| $Kernel$ | $0$ | $K_{1}$ | $K_{2}$ | $K_{3}$ | $0$ | $K_{4}$ | $K_{5}$ | $K_{6}$ | $0$ | $K_{7}$ | $K_{8}$ | $K_{9}$ | $0$ | $0$ | $0$ | $0$ |
| $Kernel$ | $0$ | $0$ | $0$ | $0$ | $K_{1}$ | $K_{2}$ | $K_{3}$ | $0$ | $K_{4}$ | $K_{5}$ | $K_{6}$ | $0$ | $K_{7}$ | $K_{8}$ | $K_{9}$ | $0$ |
| $Kernel$ | $0$ | $0$ | $0$ | $0$ | $0$ | $K_{1}$ | $K_{2}$ | $K_{3}$ | $0$ | $K_{4}$ | $K_{5}$ | $K_{6}$ | $0$ | $K_{7}$ | $K_{8}$ | $K_{9}$ |


**转置卷积**: \
假设图像尺寸为 $2 \times 2$ , 卷积核为 $3 \times 3$ , padding=0, stride=1 \
则在代码中会将图片和卷积核都转换为矩阵，转换后的矩阵形式如下：\
图像: $I_{4 \times 1}$ , 即图像会被展平 \
卷积核: $K_{16 \times 4}$，下标16表示输出有16行（这也需要根据各种size计算得来），下标4表示的是从$3 \times 3$的卷积核中取出来的4个数（如图3(b)所示，一个$3 \times 3$的卷积核与$2 \times 2$的图片最多只有4个点会接触，其他的只有1~2个接触点）\
输出: $O_{16 \times 1} = K_{16 \times 4} \cdot I_{4 \times 1}$


> 所以，关于为什么称为转置卷积？现在可以回答这个问题了：
> 对于上例，一个正常卷积的卷积核在代码中会被转换为一个$4 \times 16$的卷积核，而一个转置卷积的卷积核在代码中会被转换为$16 \times 4$，从形状上看，后者看起来像是前者的转置，所以故名“转置卷积”。\
> 但是也要注意，它们**只是形状上看起来像是转置关系**，但实际上它们的权值是不相同的，而由于权值的不同，所以**正常卷积和转置卷积是不可逆的**，也即是说一个$4 \times 4$的矩阵经过正常卷积得到了$2 \times 2$矩阵，而这个$2 \times 2$的矩阵再经过转置卷积得到$4 \times 4$的矩阵，前面的$4 \times 4$矩阵和后面的$4 \times 4$矩阵是完全不相等的。


## nn.ConvTranspose2d
```python
class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
```
**功能**：转置卷积实现上采样

- **in_channels**: 输入通道数
- **out_channels**: 输出通道数
- **kernel_size**: 卷积核尺寸
- **stide**: 步长
- **padding**: 填充个数
- **dilation**: 空洞卷积大小
- **groups**: 分组卷积设置
- **bias**: 偏置

尺寸计算：\
简化版：
$$out_{size} = (in_{size} - 1) * stride + kernel_{size}$$
可以发现，这和正常卷积的简化版计算公式恰好相反。

完整版：
$$H_{out} = (H_{in} - 1) \times stride[0] - 2 \times padding[0] + dilation[0] \times (kernel_size[0] - 1) + output_padding[0] + 1$$


代码示例：
```python
# ================ transposed
flag = 1
# flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)
```

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_conv5.jpg" width = 60% height = 60% />
</div>
<center>图4 &nbsp;nn.ConvTranspose2d转置卷积可视化与原图比较(左侧为原图，右侧为转置卷积后的效果图)</center>

如上图所示，转置卷积后的图像有个很奇怪的现象，这是转置矩阵的通病，称为**棋盘效应**，是由不均匀重叠导致的。\
关于棋盘效应的解释以及解决方法看参考：《Deconvolution and Checkerboard Artifacts》



# 参考文献
[1] DeepShare.net > PyTorch框架
