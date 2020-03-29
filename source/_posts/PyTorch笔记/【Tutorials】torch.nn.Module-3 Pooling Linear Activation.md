---
title: 
date: 2019-09-25
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述PyTorch中的Pooling Layer、Linear Layer和Activation Layer.
<!-- more -->

# Pooling Layer
池化运算: 对信号进行 “收集” 并 “总结”, 类似水池收集水资源,因而得名池化层

> “收集”:多变少 \
> “总结”:最大值/平均值

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_pooling1.jpg" width = 60% height = 60% />
</div>


## nn.MaxPool2d
&emsp; nn.MaxPool2d 继承自 _MaxPoolNd，nn.MaxPool2d 自己没有初始化函数，它使用的是 _MaxPoolNd 的初始化函数，nn.MaxPool2d 自己实现forward函数。

```python
class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.
    """
    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
```
**功能**：对二维信号(图像)进行最大池化

主要参数：
- **kernel_size**: 池化核尺寸，通常使用正方形的池化核，例如(2, 2)
- **stride**: 步长，通常和kernel_size相等，即不重叠采样。例如kernel_size=(2, 2), stride也设为(2, 2)
- **padding**: 填充个数
- **dilation**: 池化核间隔大小
- **ceil_mode**: 尺寸向上取整或者向下取整。在计算输出feature-map尺寸的时候，有一个除法操作，如果该除法不能整除时就需要采用取整操作，当ceil_mode为True时采用向上取整,而False表示采样向下取整，默认为False.
- **return_indices**: 记录池化像素索引，用于记录最大值像素所在的位置的索引，该索引通常在**最大值反池化**上采样的时候使用。

> 关于**最大值反池化**：
> 在早期的自编码器以及图像分割任务当中都会涉及上采样的操作，当时都会采用最大值反池化进行上采样:
> <div align=center><img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_pooling2.jpg" width = 60% height = 60% /></div>


## nn.AvgPool2d

**功能**：对二维信号(图像)进行平均池化

主要参数：
- **kernel_size**: 池化核尺寸，同nn.MaxPool2d
- **stride**: 步长，同nn.MaxPool2d
- **padding**: 填充个数，同nn.MaxPool2d
- **ceil_mode**: 尺寸向上取整，同nn.MaxPool2d
- **count_include_pad**: 填充值用于计算，即在计算平均值的时候是否采用填充值进行计算，如果设置为True，那么在计算平均值的时候，也会将填充值这些像素加进来进行计算。
- **divisor_override**: 除法因子。我们知道，在做池化平均的时候，是将池化核内的所有像素值相加 再除以这些像素值的个数，而通过divisor_override这个参数，我们可以改变这一模式，即不再除以像素值的个数，而是除以divisor_override.

nn.MaxPool2d、nn.AvgPool2d 代码示例：
```python
# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_others.py
# @author     : tingsongyu
# @date       : 2019-09-25 10:08:00
# @brief      : 其它网络层
"""
import os
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from tools.common_tools import transform_invert, set_seed

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ maxpool
flag = 1
# flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)

# ================ avgpool
# flag = 1
flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = avgpoollayer(img_tensor)

# ================================= visualization ==================================
print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))
img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_pool)
plt.subplot(121).imshow(img_raw)
plt.show()
```


关于 nn.AvgPool2d 的 divisor_override参数的使用代码示例：
```python
# ================ avgpool divisor_override
flag = 1
# flag = 0
if flag:
    img_tensor = torch.ones((1, 1, 4, 4))
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{}\npooling_img:\n{}".format(img_tensor, img_pool))


# output:
raw_img:
tensor([[[[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]]]])
pooling_img:
tensor([[[[1.3333, 1.3333],
          [1.3333, 1.3333]]]])
```


## nn.MaxUnpool2d
```python
class _MaxUnpoolNd(Module):

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )


class MaxUnpool2d(_MaxUnpoolNd):
    r"""Computes a partial inverse of :class:`MaxPool2d`.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool2d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)
```
**功能**：对二维信号(图像)进行最大池化上采样

主要参数：
- **kernel_size**: 池化核尺寸
- **stride**: 步长
- **padding**: 填充个数

代码示例：
```python
# ================ max unpool
flag = 1
# flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    # unpooling
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print("raw_img:\n{}\nimg_pool:\n{}".format(img_tensor, img_pool))
    print("img_reconstruct:\n{}\nimg_unpool:\n{}".format(img_reconstruct, img_unpool))


# output:
raw_img:
tensor([[[[0., 4., 4., 3.],
          [3., 3., 1., 1.],
          [4., 2., 3., 4.],
          [1., 3., 3., 0.]]]])
img_pool:
tensor([[[[4., 4.],
          [4., 4.]]]])
img_reconstruct:
tensor([[[[-1.0276, -0.5631],
          [-0.8923, -0.0583]]]])
img_unpool:
tensor([[[[ 0.0000, -1.0276, -0.5631,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000],
          [-0.8923,  0.0000,  0.0000, -0.0583],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]]])
```

# Linear Layer
线性层(Linear Layer)又称全连接层, 其每个神经元与上一层所有神经元相连实现对前一层的**线性组合**或者**叫线性变换**(如果不考虑非线性变换的话).


## nn.Linear
```python
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```
**功能**：对一维信号(向量)进行线性组合

主要参数：
- **in_features**: 输入结点数
- **out_features**: 输出结点数
- **bias**: 是否需要偏置

计算公式：$\boldsymbol{y = xW^T + bias}$

```python
# ================ linear
flag = 1
# flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)

    # 注意： linear_layer.weight 是一个[4x3]的矩阵，但是PyTorch在内部计算的时候会自动将其转换为[3x4]的矩阵进行计算。
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)


# output:
tensor([[1., 2., 3.]]) torch.Size([1, 3])
tensor([[1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.],
        [4., 4., 4.]]) torch.Size([4, 3])
tensor([[ 6.5000, 12.5000, 18.5000, 24.5000]], grad_fn=<AddmmBackward>) torch.Size([1, 4])
```

# Activation Layer
激活函数对特征进行非线性变换, 赋予多层神经网络具有**深度**的意义。为什么这么说呢？
想象一下，假如没有激活函数，设有输入 $X$，3层全连接层的权重分别为 $W_1, W_2, W_3$，则有下面的推导：
$$
\begin{aligned}
  H_1 &= X * W_1 \\
  H_2 &= H_1 * W_2 \\
  Output &= H_2 * W_3 \\
  &= H_1 * W_2 * W_3 \\
  &= X * (W_1 * W_2 * W_3) \\
  &= X * W
\end{aligned}
$$

由上述推到可知，一个3层的线性全连接层其实等价于1层线性层，这是由于线性运算的**结合律**决定的。当然上述推导过程是在没有激活函数的情况下进行的，这也反应了激活函数的重要性，如果没有激活函数，再多的全连接网络，都相当于只有一层网络。


## nn.Sigmoid
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_activation_sigmoid.jpg" width = 60% height = 60% />
</div>

sigmoid函数也称logistic函数。

计算公式：$y = \frac{1}{1 + e^{-x}}$
梯度公式：$y' = y * (1 - y)$

由于多层神经网络在计算梯度的时候采用的是链式法则，在链式法则中是要叠加的乘以激活函数的梯度，由于sigmoid的梯度取值范围是 $[0, 0.25]$，叠加相乘会导致梯度越乘越小，进而导致梯度消失。

sigmoid的分布也是非常不好的，从上图蓝色曲线可以看到sigmoid的输出值都是大于0的，也就是说sigmoid的分布都是非0均值的，这会破坏数据的0均值分布。

特性：
- 输出值在(0, 1), 符合概率
- 导数范围是[0, 0.25], 易导致梯度消失
- 输出为非0均值, 破坏数据分布


## nn.tanh
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_activation_tanh.jpg" width = 60% height = 60% />
</div>

$tanh$ 又称双曲正切函数。
计算公式：$y = \frac{sin(x)}{cos(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2x}} + 1$
梯度公式：$y' = 1 - y^2$
特性：
- 输出值在(-1, 1)之间，数据符合0均值，这一点上比sigmoid就要好一些了。
- 导数范围是(0, 1)，比sigmoid的(0, 0.25)尺度更大一些，梯度消失没sigmoid那么严重，但也比较易导致梯度消失.


## nn.ReLU
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_activation_relu.jpg" width = 60% height = 60% />
</div>

&emsp; ReLU(Rectified Linear Units)中文名称线性修正单元。\
> 为什么称之为线性修正单元？
> 因为ReLU在x的正半轴 $y = x$ 是线性的，而负半轴输出都是0(不再是线性的)，所以说它是对$y=x$这一线性关系的修正，所以称之为线性修正单元。

计算公式：$y = max(0, x)$ \
梯度公式：
$$
y = \begin{cases}
    1, \qquad &x > 0 \\
    undefined, \qquad &x = 0 \\
    0, \qquad &x < 0
\end{cases}
$$

特性：
- 输出值均为正数，负半轴导致死神经元
- 导数是1，缓解梯度消失，但是又引来了另一个问题：由于导数是1，所以在叠加相乘的时候不会改变梯度的尺度，这样叠加相乘会容易引发梯度爆炸。

## nn.LeakyReLU
- **negative_slope**: 负半轴斜率

## nn.PReLU
- **init**: 可学习斜率

## nn.RReLU
RReLU的第一个R表示Rand.

- **lower**: 均匀分布下限
- **upper**: 均匀分布上限

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module_activation_LeakyRelu.jpg" width = 60% height = 60% />
</div>


## 激活函数总结
&emsp; sigmoid、tanh都是饱和函数，会引发梯度消失，因此可以采用非饱和函数ReLU进行替换，而针对ReLU函数的负半轴带来的一系列问题，ReLU又有一系列的变种。


# 参考文献
[1] DeepShare.net > PyTorch框架