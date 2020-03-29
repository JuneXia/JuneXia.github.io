---
title: 
date: 2020-02-12
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节首先讲述什么是损失函数以及损失函数与代价函数、目标函数之间的关系，然后讲述了PyTorch中的常用损失函数。
<!-- more -->

# 什么是损失函数
**损失函数**：衡量模型输出与真实标签的差异

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_loss1.jpg" width = 80% height = 80% />
</div>

**损失函数（Loss Function）**:
$$
Loss = f(\hat{y}, y)
$$
损失函数是计算一个样本的预测值与真实值之间的差异的。

**代价函数（Cost Function）**:
$$
Cost = \frac{1}{N} \sum^N_{i} f(\hat{y_i}, y_i)
$$
代价函数是计算整个样本集（整个训练集）Loss的平均值，也就是在损失函数的基础上求均值。

**目标函数（Objective Function）**:
$$
Obj = Cost + Regularization
$$
目标函数是一个更广泛的概念，在机器学习中，目标函数通常就是训练的最终目标，通常整个目标会包括 Cost Function 和Regularization. 因为通常情况下，这个Cost并不是越小越好，太小了容易导致过拟合，所以要对机器学习做一些约束，通常称这些约束为**正则化(Regularization)**

为了不失一般性，后面通常都会用Loss Function去代替Cost Function.


PyTorch中的loss继承自Module，
```python
class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
```
- **size_average**: 即将弃用
- **reduce**: 即将弃用
- **reduction**: 


# PyTorch中的损失函数

## nn.CrossEntropyLoss
```python
class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
```
**功能**: nn.LogSoftmax() 与 nn.NLLLoss() 结合，进行交叉熵计算

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_loss2.jpg" width = 80% height = 80% />
</div>

交叉熵是用来衡量两个分布还见的差异，关于交叉熵的更多原理性的东西其他文章以及网络资源都已经讲过了，这里就不具体展开讲了。

主要参数：
- **weight**: 各类别的loss设置权值
- **ignore_index**: 忽略某个类别
- **reduction**: 计算模式，可为 none/sum/mean \
&emsp; &emsp; &emsp; &emsp; none: 逐个元素(样本)计算 \
&emsp; &emsp; &emsp; &emsp; sum: 所有元素求和，返回标量 \
&emsp; &emsp; &emsp; &emsp; mean: 加权平均，返回标量
- **size_average**: 即将弃用
- **reduce**: 即将弃用

交叉熵的计算公式为：
$$
\begin{aligned}
    H(P, Q) = - \sum^N_{i=1} P(x_i) log Q(x_i)
\end{aligned}
$$
其中 $P(x_i)$ 表示真实概率，通常为0或者1，$Q(x_i)$ 表示预测概率。

CrossEntropyLoss在程序中的计算公式：
$$
loss(x, class) = -log \Big(\frac{exp(x[class])}{\sum_j exp(x[j])} \Big) = 
-x[class] + log \Big(\sum_j exp(x[j]) \Big)
$$

考虑到权值的CrossEntropyLoss在程序中的计算公式：
$$
loss(x, class) = weight[class] \Bigg (-x[class] + log \Big(\sum_j exp(x[j]) \Big) \Bigg)
$$

考虑到权值的CrossEntropyLoss均值在程序中的计算公式：(加权平均)
$$
Mean \Big(loss(x, class)) \Big) = \frac{weight[class] \Bigg (-x[class] + log \Big(\sum_j exp(x[j]) \Big) \Bigg)}{\sum_j weight_j}
$$
注意在weight模式下，求交叉熵Mean Loss的时候除的不是样本的个数，而是除以样本权重的总和，这也可以通过后面的代码看到。


### 不带有权值的loss的代码示例
```python
# -*- coding: utf-8 -*-
"""
# @file name  : loss_function_1.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-10-07 10:08:00
# @brief      : 1. nn.CrossEntropyLoss
                2. nn.NLLLoss
                3. BCELoss
                4. BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

# ----------------------------------- CrossEntropy loss: reduction -----------------------------------
# flag = 0
flag = 1
if flag:
    # def loss function
    loss_f_none = nn.CrossEntropyLoss(weight=None, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

    # forward
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("Cross Entropy Loss:\n ", loss_none, loss_sum, loss_mean)

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    idx = 0

    input_1 = inputs.detach().numpy()[idx]      # [1, 2]
    target_1 = target.numpy()[idx]              # [0]

    # 第一项
    x_class = input_1[target_1]

    # 第二项
    sigma_exp_x = np.sum(list(map(np.exp, input_1)))
    log_sigma_exp_x = np.log(sigma_exp_x)

    # 输出loss
    loss_1 = -x_class + log_sigma_exp_x

    print("第一个样本loss为: ", loss_1)


# output:
Cross Entropy Loss:
loss_none:
  tensor([1.3133, 0.1269, 0.1269]) 

loss_sum:  
tensor(1.5671)

loss_mean:
tensor(0.5224)

手动计算：
第一个样本loss为:  1.3132617
```

### 带有权值的loss的代码示例
```python
# flag = 0
flag = 1
if flag:
    # def loss function
    weights = torch.tensor([1, 2], dtype=torch.float)
    # weights = torch.tensor([0.7, 0.3], dtype=torch.float)

    loss_f_none_w = nn.CrossEntropyLoss(weight=weights, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("\nweights: ", weights)
    print("loss_none: \n ", loss_none_w)
    print("loss_sum:\n ", loss_sum)
    print("loss_mean:\n ", loss_mean)


# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:
    weights = torch.tensor([1, 2], dtype=torch.float)
    weights_all = np.sum(list(map(lambda x: weights.numpy()[x], target.numpy())))  # [0, 1, 1]  # [1 2 2]

    mean = 0
    loss_sep = loss_none.detach().numpy()
    for i in range(target.shape[0]):

        x_class = target.numpy()[i]
        tmp = loss_sep[i] * (weights.numpy()[x_class] / weights_all)
        mean += tmp

    print("手动计算的的 loss_mean:\n ", mean)


# output:
# Pytorch 计算的带有 weight 的损失。
weights:  tensor([1., 2.])

loss_none: 
  tensor([1.3133, 0.2539, 0.2539])

loss_sum:
  tensor(1.8210)

loss_mean:
  tensor(0.3642)


# 手动计算的带有 weight 的mean损失
手动计算的的 loss_mean:
  0.3641947731375694
```


## nn.NLLLoss
```python
class NLLLoss(_WeightedLoss):
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
    this class index (this index may not necessarily be in the class range).

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When
            :attr:`size_average` is ``True``, the loss is averaged over
            non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> m = nn.LogSoftmax(dim=1)
        >>> loss = nn.NLLLoss()
        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = loss(m(input), target)
        >>> output.backward()
        >>>
        >>>
        >>> # 2D loss example (used, for example, with image inputs)
        >>> N, C = 5, 4
        >>> loss = nn.NLLLoss()
        >>> # input is of size N x C x height x width
        >>> data = torch.randn(N, 16, 10, 10)
        >>> conv = nn.Conv2d(16, C, (3, 3))
        >>> m = nn.LogSoftmax(dim=1)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        >>> output = loss(m(conv(data)), target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'weight', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
```
**功能**：实现负对数似然函数中的**负号功能**

在代码中的计算公式如下：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_loss3.jpg" width = 80% height = 80% />
</div>

其中 $w_{y_n}$ 表示对应类别的权重，$x_{n, y_n}$ 表示对应类别的预测值，如果 $w_{y_n} = 1$ 的话，相当于只是对预测结果取负号。

主要参数：
- **weight**: 各类别的loss设置权值
- **ignore_index**: 忽略某个类别
- **reduction**: 计算模式，可为 none/sum/mean \
&emsp; &emsp; &emsp; &emsp; none: 逐个元素(样本)计算 \
&emsp; &emsp; &emsp; &emsp; sum: 所有元素求和，返回标量 \
&emsp; &emsp; &emsp; &emsp; mean: 加权平均，返回标量

```python
# ----------------------------------- 2 NLLLoss -----------------------------------
# flag = 0
flag = 1
if flag:

    # 为不失一般性，这里还是将类别权重设置为1，这也就上相当于不考虑类别权重
    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.NLLLoss(weight=weights, reduction='none')
    loss_f_sum = nn.NLLLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.NLLLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("\nweights: ", weights)
    print("loss_none:\n ", loss_none_w)
    print("loss_sum:\n ", loss_sum)
    print("loss_mean:\n ", loss_mean)


# output:
weights:  tensor([1., 1.])
loss_none: 
  tensor([-1., -3., -3.])
loss_sum:
  tensor(-7.)
loss_mean:
  tensor(-2.3333)
```


## nn.BCELoss
```python
class BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets :math:`y` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    __constants__ = ['reduction', 'weight']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
```
**功能**：二分类交叉熵，是交叉熵损失函数的特例。
**注意事项**: 输入值取值在[0, 1]

计算公式：
$$
l_n = -w_n[ y_n \cdot log x_n + (1 - y_n) \cdot log(1 - x_n) ]
$$
其中，$x_n$ 是模型输出概率值，$y_n$ 是标签。

主要参数：
- **weight**: 各类别的loss设置权值
- **ignore_index**: 忽略某个类别
- **reduction**: 计算模式，可为 none/sum/mean \
&emsp; &emsp; &emsp; &emsp; none: 逐个元素(样本)计算 \
&emsp; &emsp; &emsp; &emsp; sum: 所有元素求和，返回标量 \
&emsp; &emsp; &emsp; &emsp; mean: 加权平均，返回标量


```python
# ----------------------------------- 3 BCE Loss -----------------------------------
# flag = 0
flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    # 将预测结果转化为概率取值
    inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCELoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCELoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCELoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print("BCE Loss", loss_none_w, loss_sum, loss_mean)


# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    idx = 0

    x_i = inputs.detach().numpy()[idx, idx]
    y_i = target.numpy()[idx, idx]              #

    # loss
    # l_i = -[ y_i * np.log(x_i) + (1-y_i) * np.log(1-y_i) ]      # np.log(0) = nan
    l_i = -y_i * np.log(x_i) if y_i else -(1-y_i) * np.log(1-x_i)

    # 输出loss
    print("BCE inputs: ", inputs)
    print("第一个loss为: ", l_i)



# output:
# 这里将类别权重都设置为1.
weights:  tensor([1., 1.])

# target label: 
tensor([[1., 0.],
        [1., 0.],
        [0., 1.],
        [0., 1.]])

# Pytorch 计算的 BCE Loss:
BCE Loss 
tensor([[0.3133, 2.1269],
        [0.1269, 2.1269],
        [3.0486, 0.0181],
        [4.0181, 0.0067]]) tensor(11.7856) tensor(1.4732)


# BCE inputs:  (sigmoid的输出)
tensor([[0.7311, 0.8808],
        [0.8808, 0.8808],
        [0.9526, 0.9820],
        [0.9820, 0.9933]])

# 自己手动计算的 BCE Loss
第一个loss为:  0.31326166
```


关于BCELoss的计算过程，我这里通过实例的形式再理一遍：
```python
# 假设有预测结果inputs：
tensor([[1., 2.],
        [2., 2.],
        [3., 4.],
        [4., 5.]])

# 经过sigmoid处理后的inputs:
tensor([[0.7311, 0.8808],
        [0.8808, 0.8808],
        [0.9526, 0.9820],
        [0.9820, 0.9933]])

# BCELoss需要计算inputs中每个元素的loss值，
# 例如对于 inputs[0, 0]，BCELoss的计算结果为：
loss[0, 0] = -( label[0, 0] * log(inputs[0, 0]) + (1 - label[0, 0]) * log(1 - inputs[0, 0]) )
           = -( 1 * log(0.7311) + (1 - 1) * log(1 - 0.7311) )
           = 0.31320502968286684

# 例如对于 inputs[0, 1]，BCELoss的计算结果为：
loss[0, 1] = -( label[0, 1] * log(inputs[0, 1]) + (1 - label[0, 1]) * log(1 - inputs[0, 1]) )
           = -( 0 * log(0.8808) + (1 - 0) * log(1 - 0.8808) )
           = 2.1269525243508878

loss[i, j] 这些值以此类推。

BCELoss的sum模式就是对所有的loss[i, j]求和所得。
```

## nn.BCEWithLogitsLoss
**功能**: 结合 Sigmoid 与二分类交叉熵
**注意事项**：网络最后不能加sigmoid函数，因为nn.BCEWithLogitsLoss在内部已经做了sigmoid处理，所以外面再加sigmoid函数的话就会导致输出结果不准确。

计算公式：
$$
l_n = -w_n[ y_n \cdot log \sigma(x_n) + (1 - y_n) \cdot log(1 - \sigma(x_n)) ]
$$
其中，$x_n$ 是模型输出概率值，$y_n$ 是标签，$\sigma$ 表示sigmoid函数。

主要参数：
- **pos_weight**: 正样本的权值，当正负样本不平衡时，可以用于设置正样本的权值。例如负样本有300个、正样本有有100个的时候，正负样本比例为1/3，这时候就可以设置pos_weight为3，也就是对正样本的loss乘以3，这样就等价于正样本也有300个了，从而实现了正负样本比例均衡。
- **weight**: 各类别的loss设置权值
- **ignore_index**: 忽略某个类别
- **reduction**: 计算模式，可为 none/sum/mean \
&emsp; &emsp; &emsp; &emsp; none: 逐个元素(样本)计算 \
&emsp; &emsp; &emsp; &emsp; sum: 所有元素求和，返回标量 \
&emsp; &emsp; &emsp; &emsp; mean: 加权平均，返回标量

> 由此可见，如果不考虑 pos_weight 的话，nn.BCEWithLogitsLoss 和 nn.BCELoss 实际上是等价的，因为 nn.BCELoss 需要在外面先对预测结果做sigmoid压缩处理后再输入到 nn.BCELoss 中，而 nn.BCEWithLogitsLoss 只不过是将这个sigmoid处理过程拿到了内部处理了而已。
> PyTorch 这样设计的原因可能是因为在有些场合下可能只是需要用sigmoid函数来计算一下损失而已，而其他情况下并不需要sigmoid函数。


### 不使用pos_weight的代码示例
```python
# ----------------------------------- 4 BCE with Logis Loss -----------------------------------
# flag = 0
flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print(loss_none_w, loss_sum, loss_mean)


# output: 
tensor([[0.3133, 2.1269],
        [0.1269, 2.1269],
        [3.0486, 0.0181],
        [4.0181, 0.0067]]) tensor(11.7856) tensor(1.4732)
```
输出结果和BCELoss的一模一样。


### 使用pos_weight的代码示例
```python
# --------------------------------- pos weight

# flag = 0
flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1], dtype=torch.float)
    pos_w = torch.tensor([3], dtype=torch.float)        # 3

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none', pos_weight=pos_w)
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum', pos_weight=pos_w)
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean', pos_weight=pos_w)

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\npos_weights: ", pos_w)
    print(loss_none_w, loss_sum, loss_mean)


# output:
pos_weights:  tensor([3.])
tensor([[0.9398, 2.1269],
        [0.3808, 2.1269],
        [3.0486, 0.0544],
        [4.0181, 0.0201]]) tensor(12.7158) tensor(1.5895)
```
正样本的loss值都被扩大了3被，导致总体loss值也变大了，这使得模型在训练时会更加关注正样本。




# 参考文献
[1] DeepShare.net > PyTorch框架