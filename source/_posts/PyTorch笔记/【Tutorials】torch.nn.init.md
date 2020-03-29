---
title: 
date: 2020-02-11
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节首先介绍梯度消失与梯度爆炸，然后逐步分析权值初始化的重要性，随后介绍了正对不同激活函数的初始化方法。
<!-- more -->

&emsp; 一个好的权值初始化能够加快模型的收敛，而不恰当的初始化可能引发梯度消失或爆炸，最终导致模型无法收敛。

# 梯度消失与爆炸
关于梯度消失与爆炸(Gradient Vanishing and Exploding)，我们来看看$\mathbf{W_2}$的梯度求取过程：

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_init1.jpg" width = 80% height = 80% />
</div>

$$
\begin{aligned}
\mathbf{H_2} &= \mathbf{W_2} * \mathbf{W_2} \\
\Delta \! \mathbf{W_2} &= \frac{\partial Loss}{\partial \mathbf{W_2}} 
                   = \frac{\partial Loss}{\partial out} * \frac{\partial out}{\partial \mathbf{H_2}} * \frac{\partial \mathbf{H_2}}{\partial \mathbf{w_2}} \\
                  &= \frac{\partial Loss}{\partial out} * \frac{\partial out}{\partial \mathbf{H_2}} * \mathbf{H_1}
\end{aligned} \tag{1}
$$


&emsp; 由式(1)可以看出，$\mathbf{W_2}$ 的梯度 $\Delta \\! \mathbf{W_2}$ 会依赖上一层的输出 $\mathbf{H_1}$，如果 $\mathbf{H_1} \rightarrow 0$，则 $\Delta \\! \mathbf{W_2} \rightarrow 0$，引发梯度消失；如果 $\mathbf{H_1} \rightarrow \infty$，则 $\Delta \\! \mathbf{W_2} \rightarrow \infty$，引发梯度爆炸。

梯度消失：$\mathbf{H_1} \rightarrow 0 \qquad \Rightarrow \qquad \Delta \\! \mathbf{W_2} \rightarrow 0$ \
梯度爆炸：$\mathbf{H_1} \rightarrow \infty \qquad \Rightarrow \qquad \Delta \\! \mathbf{W_2} \rightarrow \infty$

&emsp; 从公式(1)角度来看，要避免梯度消失与爆炸，就要严格控制各个网络层的输出尺度(不能太大也不能太小)。

为了便于讨论，我们先不考虑激活函数对神经网络的影响。

## 标准正态分布初始化
下面我们使用标准正态分布(0均值1标准差)来初始化权值，通过代码来验证一下梯度消失与爆炸：

```python
# -*- coding: utf-8 -*-
"""
# @file name  : grad_vanish_explod.py
# @author     : tingsongyu
# @date       : 2019-09-30 10:08:00
# @brief      : 梯度消失与爆炸实验
"""
import os
import torch
import random
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 标准正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                nn.init.normal_(m.weight.data)  # 标准正态分布 normal: mean=0, std=1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# flag = 0
flag = 1

if flag:
    layer_nums = 100
    neural_nums = 256
    batch_size = 16

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

    output = net(inputs)
    print(output)


# output:
tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], grad_fn=<MmBackward>)
```

发现上述代码的输出是 nan，也就是说我们的数据非常大或者非常小，已经超出了当前精度可表示的范围。
为了分析在网络的哪一层开始就出现了 nan，我们在forward函数中增加一些打印代码，这里我们使用标准差来衡量数据的尺度范围。

```python
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        ...

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x
    
    ...


# output:
layer:0, std:15.959932327270508
layer:1, std:256.6237487792969
layer:2, std:4107.24560546875
layer:3, std:65576.8125
layer:4, std:1045011.875
layer:5, std:17110406.0
layer:6, std:275461408.0
layer:7, std:4402537472.0
...
layer:29, std:1.322983152787379e+36
layer:30, std:2.0786817918687285e+37
layer:31, std:nan
output is nan in 31 layers
tensor([[        inf, -2.6817e+38,         inf,  ...,         inf,
                 inf,         inf],
        [       -inf,        -inf,  1.4387e+38,  ..., -1.3409e+38,
         -1.9660e+38,        -inf],
        [-1.5873e+37,         inf,        -inf,  ...,         inf,
                -inf,  1.1484e+38],
        ...,
        [ 2.7754e+38, -1.6783e+38, -1.5531e+38,  ...,         inf,
         -9.9440e+37, -2.5132e+38],
        [-7.7183e+37,        -inf,         inf,  ..., -2.6505e+38,
                 inf,         inf],
        [        inf,         inf,        -inf,  ...,        -inf,
                 inf,  1.7432e+38]], grad_fn=<MmBackward>)
```

发现网络在layer31就已经出现了 nan。


## 通过方差来分析为什么会出现梯度消失与爆炸
下面我们通过方差的公式推导来观察网络层的输出标准差为什么会越来越大，最后超出了我们的表示范围。

在进行方差公式推导之前，我们先复习一下关于方差的3个基本公式。

1. $E(X \* Y) = E(X) \* E(Y)$，两个相互独立的随机变量乘积的期望等于它们各自期望的乘积
2. $D(X) = E(X^2) - [E(X)]^2$
3. $D(X + Y) = D(X) + D(Y)$，两个相互独立的随机变量之和的方差等于它们各自方差之和
4. 1.2.3 $\Rightarrow D(X \* Y) = D(X) \* D(Y) + D(X) \* [E(Y)]^2 + D(Y) \* [E(X)]^2$ \
&emsp; &emsp; &emsp; 若 $E(X) = 0, E(Y) = 0$ \
&emsp; &emsp; &emsp; 则 $D(X \* Y) = D(X) \* D(Y)$

以第一个隐藏层的第一个输出 $H_{11}$ 为例，来分析网络层的标准差是如何变化的。

$$
\begin{aligned}
    H_{11} &= \sum^n_{i=0} D(X_i) * D(W_{1i}) \\
    因为 D(X * Y) &= D(X) * D(Y) \\
    所以 D(H_{11}) &= \sum^n_{i=0} D(X_i) * D(W_{1i})
\end{aligned}
$$

由于输入 $X$ 是服从0均值1标准差的，所以 $D(X) = 1$, 而W也被设置成标准正态分布(0均值1标准差)，所以 $D(W1) = 1$，所以有：
$$
\begin{aligned}
    D(H_{11}) &= n * (1 * 1) \\
    &= n
\end{aligned} \tag{2}
$$

则标准差： $std(H_{11}) = \sqrt{D(H_{11})} = \sqrt{n}$ \
相应的 $std(H_{12})、std(H_{13})、... 、std(H_{1m})$ 也都等于 $\sqrt{n}$

> &emsp; 从公式(2)可知，$D(H_{11})$ 主要由上一层神经元的个数n、上一层输出值的方差 $D(X)$ 以及当前层 $W$ 的方差所决定的，同理其他层的其他神经元也是类似的。

&emsp; 从公式推导我们发现，对于输入方差为1的数据，经过一层前向网络的传播，第一个隐藏层输出值的方差就变成了n，即方差扩大了n倍，标准差扩大了 $\sqrt{n}$ 倍，如果再往下传播到第二个隐藏层的时候，同理可推知其输出值方差变为 $n*(n*1)=n^2$，标准差就变成了 $\sqrt{n^2} = n$. \
&emsp; 综上所述，每往后传播一层，输出值的标准差都会在前一层的基础上扩大 $\sqrt{n}$ 倍，也就是说每一层的输出值的尺度扩大了 $\sqrt{n}$ 倍，最终超出了精度可表示的范围，引发 nan.

> 通过上述代码的输出，也可以看出每一层的输出值的标准差都会扩大 $\sqrt{n}$ 倍：\
> 上述代码中 n = 256 \
> layer:0, std:15.959932327270508 = $\sqrt{256}$ \
> layer:1, std:256.6237487792969 = $\sqrt{256} \* \sqrt{256}$ \
> layer:2, std:4107.24560546875 = $\sqrt{256} \* \sqrt{256} \* \sqrt{256}$ \
> ...

## 改进的正态分布初始化
&emsp; 通过公式(2)我们发现，如果要想让每一层输出值的方差保持尺度不变，那么只能让方差等于1，也就是让下式成立：
$$D(\mathbf{H_1}) = n * D(\mathbf{X}) * D(\mathbf{W}) = 1$$
也就是要求：
$$D(\mathbf{W}) = \frac{1}{n} \quad \Rightarrow \quad std(\mathbf{W}) = \sqrt{\frac{1}{n}}$$

根据这个思想，我们将代码中每层weight的标准差都初始化为 $\sqrt{\frac{1}{n}}$.

```python
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        ...

    def forward(self, x):
        ...

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 标准正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data)  # 标准正态分布 normal: mean=0, std=1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 改进后正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# output:
layer:0, std:0.9974957704544067
layer:1, std:1.0024365186691284
layer:2, std:1.002745509147644
...
layer:98, std:1.1617801189422607
layer:99, std:1.2215301990509033
tensor([[-1.0696, -1.1373,  0.5047,  ..., -0.4766,  1.5904, -0.1076],
        [ 0.4572,  1.6211,  1.9660,  ..., -0.3558, -1.1235,  0.0979],
        [ 0.3909, -0.9998, -0.8680,  ..., -2.4161,  0.5035,  0.2814],
        ...,
        [ 0.1876,  0.7971, -0.5918,  ...,  0.5395, -0.8932,  0.1211],
        [-0.0102, -1.5027, -2.6860,  ...,  0.6954, -0.1858, -0.8027],
        [-0.5871, -1.3739, -2.9027,  ...,  1.6734,  0.5094, -0.9986]],
       grad_fn=<MmBackward>)
```

可以看到现在每一层的标准差都在1附近，且最后一层的输出值也都在正常范围内。

## 有激活函数的神经网络的初始化
但是上面的推导还没有考虑到激活函数的存在，下面我们将代码中每一层的输出都加上一个激活函数，然后看看会出现什么现象。

```python
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        ...
    
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.tanh(x)  # 每一层输出增加一个激活函数

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x
    
    def initialize(self):
        ...


# output:
layer:0, std:0.6273701786994934
layer:1, std:0.48910173773765564
layer:2, std:0.4099564850330353
...
layer:98, std:0.07842092216014862
layer:99, std:0.08206240087747574
tensor([[-0.1103, -0.0739,  0.1278,  ..., -0.0508,  0.1544, -0.0107],
        [ 0.0807,  0.1208,  0.0030,  ..., -0.0385, -0.1887, -0.0294],
        [ 0.0321, -0.0833, -0.1482,  ..., -0.1133,  0.0206,  0.0155],
        ...,
        [ 0.0108,  0.0560, -0.1099,  ...,  0.0459, -0.0961, -0.0124],
        [ 0.0398, -0.0874, -0.2312,  ...,  0.0294, -0.0562, -0.0556],
        [-0.0234, -0.0297, -0.1155,  ...,  0.1143,  0.0083, -0.0675]],
       grad_fn=<TanhBackward>)
```

可以看到，网络各层的标准差随着前向传播的进行变得越来越小，也就是数据会变得越来越小，从而导致梯度消失，而这并不是我们希望看到的现象。

# Xavier方法与Kaiming方法

## Xavier初始化：针对饱和激活函数

&emsp; 针对具有激活函数的神经网络该如何初始化，2010年，Xavier发表了一篇文章《Understanding the difficulty of training deep feedforward neural networks》详细探讨了这个问题。在论文中，结合方差一致性原则，也就是让网络每一层输出值的方差都尽量等于1，同时针对饱和激活函数(如Sigmoid、Tanh)进行分析。

> **方差一致性**：保持数据尺度维持在恰当范围，通常方差为1.

通过论文中的公式推导，可以得到权值 $\mathbf{W}$ 应该满足下面两个等式：
$$
\begin{aligned}
    n_i * D(\mathbf{W}) &= 1 \\
    n_{i + 1} * D(\mathbf{W}) &= 1
\end{aligned}
$$

这里 $n_i$ 表示输入层(上一层)的神经元个数，$n_{i+1}$ 表示输出层(当前层)的神经元个数，这是同时考虑了前向传播和反向传播的输出尺度问题而得到的两个等式，同时结合方差一致性原则，最终就得到了：
$$
D(\mathbf{W}) = \frac{1}{n_i + n_{i+1}}  \tag{3}
$$

&emsp; 前面说的都是对 $\mathbf{W}$ 采用标准正态分布进行初始化(即0均值1标准差)，而 Xavier 通常采用的是均匀分布，下面开始推导均匀分布的上下限。\
设 $\mathbf{W}$ 服从均匀分布：
$$W \sim U[-a, \ a]$$

> 因为通常采用0均值，所以上下限是对称的，即 $-a$ 和 $a$ 是对称的。

由均匀分布的方差公式可得：
$$
D(\mathbf{W}) = \frac{(-a - a)^2}{12} 
            = \frac{(2a)^2}{12} 
            = \frac{a^2}{3}  \tag{4}
$$

我们希望公式(3)和公式(4)应该相等，即：
$$
\begin{aligned}
    \frac{2}{n_i + n_{i+1}} = \frac{a^2}{3} 
    \quad \Rightarrow \quad 
    a = \frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}
\end{aligned}
$$

于是就得到了 $\mathbf{W}$ 所服从的均匀分布，也就是 Xavier 初始化：

$$
\mathbf{W} \sim U \Big[-\frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}, \ \ \frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}} \Big]  \tag{5}
$$

下面将在代码中使用 Xavier 初始化。
```python
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        ...
    
    def forward(self, x):
        ...

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 标准正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data)  # 标准正态分布 normal: mean=0, std=1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 改进后正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 使用自定义的 Xavier 初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                a = np.sqrt(6 / (self.neural_num + self.neural_num))

                # tanh激活函数的增益
                tanh_gain = nn.init.calculate_gain('tanh')
                a *= tanh_gain

                nn.init.uniform_(m.weight.data, -a, a)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# output:
layer:0, std:0.7571136355400085
layer:1, std:0.6924336552619934
layer:2, std:0.6677976846694946
...
layer:98, std:0.6407693028450012
layer:99, std:0.6442864537239075
tensor([[ 0.1031,  0.1310,  0.8196,  ...,  0.9400, -0.6374,  0.5231],
        [-0.9587, -0.2373,  0.8548,  ..., -0.2302,  0.9325,  0.0123],
        [ 0.9490, -0.2336,  0.8702,  ..., -0.9591,  0.7902,  0.6200],
        ...,
        [ 0.7191,  0.0887, -0.4353,  ..., -0.9587,  0.2494,  0.5407],
        [-0.9583,  0.5227, -0.8054,  ..., -0.4229, -0.6074,  0.9681],
        [ 0.6117,  0.3952,  0.1042,  ...,  0.3919, -0.5273,  0.0751]],
       grad_fn=<TanhBackward>)
```

使用 PyTorch 自带的 Xavier 初始化：
```python
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 标准正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data)  # 标准正态分布 normal: mean=0, std=1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 改进后正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 使用自定义的 Xavier 初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))

                # tanh激活函数的增益
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                #
                # nn.init.uniform_(m.weight.data, -a, a)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Pytorch自带的Xvaier初始化函数
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# output:
layer:0, std:0.7571136355400085
layer:1, std:0.6924336552619934
layer:2, std:0.6677976846694946
...
layer:98, std:0.6407693028450012
layer:99, std:0.6442864537239075
tensor([[ 0.1031,  0.1310,  0.8196,  ...,  0.9400, -0.6374,  0.5231],
        [-0.9587, -0.2373,  0.8548,  ..., -0.2302,  0.9325,  0.0123],
        [ 0.9490, -0.2336,  0.8702,  ..., -0.9591,  0.7902,  0.6200],
        ...,
        [ 0.7191,  0.0887, -0.4353,  ..., -0.9587,  0.2494,  0.5407],
        [-0.9583,  0.5227, -0.8054,  ..., -0.4229, -0.6074,  0.9681],
        [ 0.6117,  0.3952,  0.1042,  ...,  0.3919, -0.5273,  0.0751]],
       grad_fn=<TanhBackward>)
```

## 关于激活函数增益

> **激活函数增益**，这个增益的概念是指数据输入到激活函数之后标准差的变化:
> $$gain = \frac{std(\mathbf{X})}{std(tanh(\mathbf{X}))}$$
> $std(\mathbf{X})$ 是输入数据 $\mathbf{X}$ 的标准差，$std(tanh(\mathbf{X}))$ 是输入数据经过 tanh 激活函数后的输出数据的标准差。
> 
> 关于激活函数的增益，老师讲的并不详细，这里补充一些我自己的理解：\
> 激活函数 tanh 就像是一个非线性系统，输入 X 经过这个系统之后得到的输出值的数据分布相对输入数据分布肯定会发生一些变化，从标准差来看就是输出标准差相对输入标准差会不一样。由于 tanh 的特性，输出数据的标准差相对输入数据会有所减小，即 tanh 这个非线性系统将输入的数据变得更加集中了些，这一点从 tanh 的s型曲线压缩数据的特性也能看出来。
> 
> 但是上述代码中，为什么将均匀分布的上下限又向外扩充 tanh_gain 倍呢？

PyTorch中提供的计算增益的方法为：
torch.nn.init.calculate_gain
```python
def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
```
**主要功能**：计算激活函数的方差变化尺度
主要参数：
- **nonlinearity**: 激活函数名称
- **param**: 激活函数的参数，如Leaky ReLU的negative_slop


关于激活函数增益，下面也给出一个代码示例：
```python
# flag = 0
flag = 1

if flag:

    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)


# output:
gain: 1.5982500314712524
tanh_gain in PyTorch: 1.6666666666666667
```
从代码输出结果发现，我们自己计算的激活增益和PyTorch计算的激活增益几乎是相等的。


## Kaiming初始化：针对非饱和激活函数
&emsp; 虽然2010年Xavier针对Sigmoid、tanh这一类的饱和激活函数提出了有效的初始化方法，但是自2010年的AlexNet出现之后，非饱和激活函数ReLU被广泛使用，由于非饱和函数的性质，Xavier初始化方法不再适用，下面给出代码示例以说明之：

```python
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            # x = torch.tanh(x)
            x = torch.relu(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Pytorch自带的Xvaier初始化函数
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# output:
layer:0, std:0.9689465165138245
layer:1, std:1.0872339010238647
layer:2, std:1.2967971563339233
layer:3, std:1.4487521648406982
...
layer:98, std:6797727.5
layer:99, std:7640645.5
tensor([[       0.0000,  3028695.5000, 12379588.0000,  ...,
          3593894.2500,        0.0000, 24658908.0000],
        [       0.0000,  2758809.7500, 11016995.0000,  ...,
          2970420.5000,        0.0000, 23173856.0000],
        [       0.0000,  2909410.2500, 13117430.0000,  ...,
          3867128.7500,        0.0000, 28463468.0000],
        ...,
        [       0.0000,  3913274.7500, 15489629.0000,  ...,
          5777740.0000,        0.0000, 33226524.0000],
        [       0.0000,  3673706.7500, 12739651.0000,  ...,
          4193523.2500,        0.0000, 26862460.0000],
        [       0.0000,  1913917.0000, 10243700.0000,  ...,
          4573404.0000,        0.0000, 22720538.0000]],
       grad_fn=<ReluBackward0>)
```

从代码输出可以看出，对于ReLU激活函数，Xavier初始化方法会使网络层输出标准差逐层变大。

针对这一问题，2015年何凯明等人发表了一篇文章《Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification》，提出了解决方法。Kaiming初始化针对的是ReLU及其变种激活函数。

在论文中，通过公式推导就可以知道，对于ReLU激活函数，网络层的权值$\mathbf{W}$的方差就应该满足：
$$
D(\mathbf{W}) = \frac{2}{n_i}  \tag{6}
$$

对于ReLU的变种激活函数，$\mathbf{W}$的方差就应该满足：
$$
D(\mathbf{W}) = \frac{2}{(1+a^2)*n_i}  \tag{7}
$$

其中 $n_i$ 表示输入层神经元的个数，$a$ 表示负半轴的斜率。
当式(7)中的 $a=0$ 时，就得到了式(6)，因此式(6)可以看成是式(7)的特殊形式。

因此，权值 $\mathbf{W}$ 的标准差应该满足：
$$
std(\mathbf{W}) = \sqrt{\frac{2}{(1+a^2)*n_i}}
$$

下面在代码中使用Kaiming初始化：
```python
    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            # x = torch.tanh(x)
            x = torch.relu(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 标准正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data)  # 标准正态分布 normal: mean=0, std=1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 改进后正态分布初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 使用自己实现的 Xavier 初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                #
                # # tanh激活函数的增益
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                #
                # nn.init.uniform_(m.weight.data, -a, a)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Pytorch自带的Xvaier初始化函数
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # tanh_gain = nn.init.calculate_gain('tanh')
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 使用自己实现的Kaiming初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # 使用Pytorch自带的Kaiming初始化
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                nn.init.kaiming_normal_(m.weight.data)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# output:
layer:0, std:0.826629638671875
layer:1, std:0.8786815404891968
layer:2, std:0.9134422540664673
...
layer:98, std:0.6579315066337585
layer:99, std:0.6668476462364197
tensor([[0.0000, 1.3437, 0.0000,  ..., 0.0000, 0.6444, 1.1867],
        [0.0000, 0.9757, 0.0000,  ..., 0.0000, 0.4645, 0.8594],
        [0.0000, 1.0023, 0.0000,  ..., 0.0000, 0.5148, 0.9196],
        ...,
        [0.0000, 1.2873, 0.0000,  ..., 0.0000, 0.6454, 1.1411],
        [0.0000, 1.3589, 0.0000,  ..., 0.0000, 0.6749, 1.2438],
        [0.0000, 1.1807, 0.0000,  ..., 0.0000, 0.5668, 1.0600]],
       grad_fn=<ReluBackward0>)
```


# 总结：PyTorch中的10种初始化方法
PyTorch提供了10种权值初始化方法，这10种初始化方法又可以分位4大类，分别为：
> 1. Xavier 均匀分布
> 2. Xavier 正态分布


> 3. Kaiming 均匀分布
> 4. Kaiming 正态分布


> 5. 均匀分布
> 6. 正态分布
> 7. 常数分布


> 8. 正交矩阵初始化
> 9.  单位矩阵初始化
> 10. 稀疏矩阵初始化

在实际开发中具体选择哪种还要具体问题具体对待，但无论选择哪种方法都要遵循**方差一致性原则**，尽量保证每一层输出值得方差都是1.



# 参考文献
[1] DeepShare.net > PyTorch框架