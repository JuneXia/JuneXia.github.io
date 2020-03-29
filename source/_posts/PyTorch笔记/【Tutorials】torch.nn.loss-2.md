---
title: 
date: 2020-02-12
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要介绍PyTorch中的剩下的14种损失函数，加上前文的4种loss，一共是18种。
<!-- more -->

## nn.L1Loss
```python
class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
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
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)
```
**功能**：计算inputs与target之差的绝对值，在数据回归中常用。

计算公式：
$$
l_n = \vert x_n - y_n \vert
$$

## nn.MSELoss
```python
class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
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

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)
```
**功能**：计算inputs与target之差的平方，在数据回归中常用。

计算公式：
$$
l_n = (x_n - y_n)^2
$$

主要参数：
- **reduction**: 计算模式，可为 none/sum/mean \
&emsp; &emsp; &emsp; &emsp; none: 逐个元素(样本)计算 \
&emsp; &emsp; &emsp; &emsp; sum: 所有元素求和，返回标量 \
&emsp; &emsp; &emsp; &emsp; mean: 加权平均，返回标量 \

L1Loss、MSELoss 代码示例：
```python
# -*- coding: utf-8 -*-
"""
# @file name  : loss_function_2.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-10-10 10:08:00
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子

# ------------------------------------------------- 5 L1 loss ----------------------------------------------
# flag = 0
flag = 1
if flag:

    inputs = torch.ones((2, 2))
    target = torch.ones((2, 2)) * 3

    loss_f = nn.L1Loss(reduction='none')
    loss = loss_f(inputs, target)

    print("input:{}\ntarget:{}\nL1 loss:{}".format(inputs, target, loss))

# ------------------------------------------------- 6 MSE loss ----------------------------------------------

    loss_f_mse = nn.MSELoss(reduction='none')
    loss_mse = loss_f_mse(inputs, target)

    print("MSE loss:{}".format(loss_mse))


# output:
input:tensor([[1., 1.],
        [1., 1.]])
target:tensor([[3., 3.],
        [3., 3.]])
L1 loss:tensor([[2., 2.],
        [2., 2.]])
MSE loss:tensor([[4., 4.],
        [4., 4.]])
```


## nn.SmoothL1Loss
```python
class SmoothL1Loss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick).
    Also known as the Huber loss:

    .. math::
        \text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}

    where :math:`z_{i}` is given by:

    .. math::
        z_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < 1 \\
        |x_i - y_i| - 0.5, & \text{otherwise }
        \end{cases}

    :math:`x` and :math:`y` arbitrary shapes with a total of :math:`n` elements each
    the sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if sets ``reduction = 'sum'``.

    Args:
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
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N, *)`, same shape as the input

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, reduction=self.reduction)
```
**功能**：平滑的 L1Loss

主要参数：
- **reduction**: 计算模式，可为 none/sum/mean，该参数和之前讲过的Loss类似，本文就不再赘述了。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_loss4.jpg" width = 80% height = 80% />
</div>

计算公式：
$$
loss(x, y) = \frac{1}{n} \sum_i z_i
$$

$$ z_i = 
\begin{cases}
    0.5(x_i - y_i)^2, \qquad if \vert x_i - y_i \vert < 1 \\
    \vert x_i - y_i \vert - 0.5, \qquad otherwise
\end{cases}
$$
采用这种平滑的Loss可以减轻离群点带来的影响。

下面在代码中画出 L1Loss 和 SmoothL1Loss 的曲线对比：
```python
# ------------------------------------------------- 7 Smooth L1 loss ----------------------------------------------
# flag = 0
flag = 1
if flag:
    inputs = torch.linspace(-3, 3, steps=500)
    target = torch.zeros_like(inputs)

    loss_f = nn.SmoothL1Loss(reduction='none')

    loss_smooth = loss_f(inputs, target)

    loss_l1 = np.abs(inputs.numpy())

    plt.plot(inputs.numpy(), loss_smooth.numpy(), label='Smooth L1 Loss')
    plt.plot(inputs.numpy(), loss_l1, label='L1 loss')
    plt.xlabel('x_i - y_i')
    plt.ylabel('loss value')
    plt.legend()
    plt.grid()
    plt.show()
```
代码绘图结果就是上文贴出的图。


## nn.PoissonNLLLoss
```python
class PoissonNLLLoss(_Loss):
    r"""Negative log likelihood loss with Poisson distribution of target.

    The loss can be described as:

    .. math::
        \text{target} \sim \mathrm{Poisson}(\text{input})

        \text{loss}(\text{input}, \text{target}) = \text{input} - \text{target} * \log(\text{input})
                                    + \log(\text{target!})

    The last term can be omitted or approximated with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.

    Args:
        log_input (bool, optional): if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target}*\text{input}`, if ``False`` the loss is
            :math:`\text{input} - \text{target}*\log(\text{input}+\text{eps})`.
        full (bool, optional): whether to compute full loss, i. e. to add the
            Stirling approximation term

            .. math::
                \text{target}*\log(\text{target}) - \text{target} + 0.5 * \log(2\pi\text{target}).
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input = False`. Default: 1e-8
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

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> output = loss(log_input, target)
        >>> output.backward()

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`,
          the same shape as the input
    """
    __constants__ = ['log_input', 'full', 'eps', 'reduction']

    def __init__(self, log_input=True, full=False, size_average=None,
                 eps=1e-8, reduce=None, reduction='mean'):
        super(PoissonNLLLoss, self).__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input, target):
        return F.poisson_nll_loss(log_input, target, log_input=self.log_input, full=self.full,
                                  eps=self.eps, reduction=self.reduction)
```
**功能**: 泊松分布的负对数似然损失函数，是输出分类服从泊松分布的时候使用的损失函数。

主要参数:
- **log_input**: 输入是否为对数形式, 决定计算公式
- **full**: 计算所有loss, 默认为False
- **eps**: 修正项, 如果input是0或过小则容易导致nan，而eps的使用就是为了避免log(input)为nan
- **reduction**: 

**计算公式**:
log_input = True
$$
\text{loss(input, target) = exp(input) - target * input}  \tag{1}
$$

log_input = False
$$
\text{loss(input, target) = input - target * log(input + eps)}  \tag{2}
$$

通过观察公式可知，(2)式相当于是对(1)式的对应项取对数：
$$
\begin{aligned}
    \text{loss(input, target)} &= \text{log exp(input) - target * log(input + eps)} \\
    &= \text{input - target * log(input + eps)}
\end{aligned}
$$

nn.PoissonNLLLoss 会逐个神经元计算loss

代码示例：
```python
# ------------------------------------------------- 8 Poisson NLL Loss ----------------------------------------------
# flag = 0
flag = 1
if flag:

    inputs = torch.randn((2, 2))
    target = torch.randn((2, 2))

    loss_f = nn.PoissonNLLLoss(log_input=True, full=False, reduction='none')
    loss = loss_f(inputs, target)
    print("input:{}\ntarget:{}\nPoisson NLL loss:{}".format(inputs, target, loss))

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    idx = 0

    loss_1 = torch.exp(inputs[idx, idx]) - target[idx, idx]*inputs[idx, idx]

    print("第一个元素loss:", loss_1)


# output:
input:
tensor([[0.6614, 0.2669],
        [0.0617, 0.6213]])

target:
tensor([[-0.4519, -0.1661],
        [-1.5228,  0.3817]])

Poisson NLL loss:
tensor([[2.2363, 1.3503],
        [1.1575, 1.6242]])

第一个元素loss: 
tensor(2.2363)
```


## nn.KLDivLoss
```python
class KLDivLoss(_Loss):
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain
    *log-probabilities* and is not restricted to a 2D Tensor.
    The targets are given as *probabilities* (i.e. without taking the logarithm).

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        l(x,y) = L = \{ l_1,\dots,l_N \}, \quad
        l_n = y_n \cdot \left( \log y_n - x_n \right)

    where the index :math:`N` spans all dimensions of ``input`` and :math:`L` has the same
    shape as ``input``. If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';} \\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    In default :attr:`reduction` mode ``'mean'``, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. ``'batchmean'`` mode gives the correct KL divergence where losses
    are averaged over batch dimension only. ``'mean'`` mode's behavior will be changed to the same as
    ``'batchmean'`` in the next major release.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
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
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied.
            ``'batchmean'``: the sum of the output will be divided by batchsize.
            ``'sum'``: the output will be summed.
            ``'mean'``: the output will be divided by the number of elements in the output.
            Default: ``'mean'``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. note::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
        In the next major release, ``'mean'`` will be changed to be the same as ``'batchmean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, *)`,
          the same shape as the input

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.kl_div(input, target, reduction=self.reduction)
```
**功能**: 计算 KLD(divergence), KL 散度, 相对熵，用于衡量两个分布之间的相似性，也可以理解为计算两个分布的距离。
**注意事项**: 需提前将输入计算 log-probabilities, 如通过nn.logsoftmax()

KL散度定义计算公式：
$$
\begin{aligned}
    D_{KL}(P \Vert Q) = E_{x \sim p} \Bigg [ log \frac{P(x)}{Q(x)} \Bigg] &= E_{x \sim p} [log P(x) - log Q(x)] \\
    &= \sum^N_{i=1} P(X_i) (log P(X_i) - log Q(X_i))
\end{aligned}
$$
其中 P 是真实分布，$P(X_i)$ 其实也就是标签，Q 是预测分布，$Q(X_i)$ 表示的模型输出的概率。

PyTorch中的计算公式：(以一个样本为例，所以这里忽略了求和符号 $\sum$)
$$
l_n = y_n \cdot (log y_n - x_n)
$$

&emsp; KL散度在PyTorch中的计算公式与KL散度定义的计算公式在外观上看略有不同，主要表现在定义的计算公式的最后一项是减 $log Q(X_i)$，而PyTorch中的计算公式的最后一项是减 $x_n$。但实际上它们是相同的，因为PyTorch 的 nn.KLDivLoss 要求先对输入计算一个 log-probabilities，然后再送入到 nn.KLDivLoss 计算，可以通过 nn.logsoftmax() 来计算 log-probabilities.

主要参数:
- **reduction**: none/sum/mean/batchmean. 注意这里的 reduction 比前面其他 Loss 函数的 reduction 多了一个 batchmean。注意：mean的分母是 (batchsize * num_class)，而batchmean的分母是batchsize，参考下面的代码理解.
- **batchmean**: batchsize维度求平均值。

代码示例：
```python
# ------------------------------------------------- 9 KL Divergence Loss ----------------------------------------------
# flag = 0
flag = 1
if flag:

    inputs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    inputs_log = torch.log(inputs)
    target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2]], dtype=torch.float)

    loss_f_none = nn.KLDivLoss(reduction='none')
    loss_f_mean = nn.KLDivLoss(reduction='mean')
    loss_f_bs_mean = nn.KLDivLoss(reduction='batchmean')

    loss_none = loss_f_none(inputs, target)
    loss_mean = loss_f_mean(inputs, target)
    loss_bs_mean = loss_f_bs_mean(inputs, target)

    print("loss_none:\n{}\nloss_mean:\n{}\nloss_bs_mean:\n{}".format(loss_none, loss_mean, loss_bs_mean))

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    idx = 0

    loss_1 = target[idx, idx] * (torch.log(target[idx, idx]) - inputs[idx, idx])

    print("第一个元素loss:", loss_1)


# output:
loss_none:
tensor([[-0.5448, -0.1648, -0.1598],
        [-0.2503, -0.4597, -0.4219]])

loss_mean:
-0.3335360586643219

loss_bs_mean:
-1.000608205795288

第一个元素loss: 
tensor(-0.5448)
```

上面代码中，inputs batchsize = 2, num_class = 3
$$
nn.KLDivLoss(reduction='mean') = \frac{sum(loss\_none)}{batchsize * num\_class}
$$

$$
nn.KLDivLoss(reduction='batchmean') = \frac{sum(loss\_none)}{batchsize * num\_class}
$$



## nn.MarginRankingLoss
```python
class MarginRankingLoss(_Loss):
    r"""Creates a criterion that measures the loss given
    inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        \text{loss}(x, y) = \max(0, -y * (x1 - x2) + \text{margin})

    Args:
        margin (float, optional): Has a default value of :math:`0`.
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
        - Input: :math:`(N, D)` where `N` is the batch size and `D` is the size of a sample.
        - Target: :math:`(N)`
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(MarginRankingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)
```
**功能**: 计算两个向量之间的相似度 ,用于排序任务 \
> **特别说明**: 该方法计算两组数据之间的差异, 返回一个 n*n 的 loss 矩阵。\
先计算第1组的第1个元素和第2组的n个元素的差异，得到n个loss值；\
再计算第1组的第2个元素和第2组的n个元素的差异，得到n个loss值；\
... \
再计算第1组的第n个元素和第2组的n个元素的差异，得到n个loss值；\
以上共得到n*n个loss值。

计算公式：
$$
loss(x, y) = max(0, -y * (x1 - x2) + margin), \qquad y \in \{-1, 1\}
$$

主要参数:
- **margin**: 边界值, x1与x2之间的差异值
- **reduction**: 计算模式, 可为none/sum/mean 

y = 1 时, 希望 x1 比 x2 大, 当 x1>x2 时, 不产生 loss （此时loss = 0）\
y = -1 时, 希望 x2 比 x1 大, 当 x2>x1 时, 不产生 loss

```python
# ---------------------------------------------- 10 Margin Ranking Loss --------------------------------------------
# flag = 0
flag = 1
if flag:

    x1 = torch.tensor([[1], [2], [3]], dtype=torch.float)
    x2 = torch.tensor([[2], [2], [2]], dtype=torch.float)

    target = torch.tensor([1, 1, -1], dtype=torch.float)

    loss_f_none = nn.MarginRankingLoss(margin=0, reduction='none')

    loss = loss_f_none(x1, x2, target)

    print(loss)



# output:
tensor([[1., 1., 0.],
        [0., 0., 0.],
        [0., 0., 1.]])
```

计算过程解析：
$$
\begin{aligned}
    output[0, 0] &= max \Big(0, -y[0] * (x1[0] - x2[0]) \Big) \\
                 &= max \Big(0, -1 * (1 - 2) \Big) = 1
\end{aligned}
$$

$$
\begin{aligned}
    output[0, 1] &= max \Big(0, -y[1] * (x1[0] - x2[1]) \Big) \\
                 &= max \Big(0, -1 * (1 - 2) \Big) = 1
\end{aligned}
$$

$$
\begin{aligned}
    output[0, 2] &= max \Big(0, -y[2] * (x1[0] - x2[1]) \Big) \\
                 &= max \Big(0, -(-1) * (1 - 2) \Big) = 0
\end{aligned}
$$


## nn.MultiLabelMarginLoss
```python
class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
    :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
    :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
    and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.

    :math:`y` and :math:`x` must have the same size.

    The criterion only considers a contiguous block of non-negative targets that
    starts at the front.

    This allows for different samples to have variable amounts of target classes.

    Args:
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
        - Input: :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C`
          is the number of classes.
        - Target: :math:`(C)` or :math:`(N, C)`, label targets padded by -1 ensuring same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    Examples::

        >>> loss = nn.MultiLabelMarginLoss()
        >>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
        >>> # for target y, only consider labels 3 and 0, not after label -1
        >>> y = torch.LongTensor([[3, 0, -1, 1]])
        >>> loss(x, y)
        >>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
        tensor(0.8500)

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MultiLabelMarginLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.multilabel_margin_loss(input, target, reduction=self.reduction)
```
**功能**: 多标签边界损失函数，不要与多分类任务混淆。

主要参数:
- **reduction**: 计算模式, 可为none/sum/mean

**什么是多标签任务**：多标签任务是指某个样本可以是多个类别。例如图片类别分类（美食类，风景类，人像类，...），对于下面的图片，它既具有“云”的标签，也具有“树”的标签，还具有“草地”标签。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_loss5.jpg" width = 60% height = 60% />
</div>

计算公式：
$$
\begin{aligned}
    loss(x, y) &= \sum_{ij} \frac{ \text{max} \Bigg( 0, 1 - \Big(x \Big[ y[j] \Big] - x[i] \Big) \Bigg) }{x.size(0)} \\
\text{where } i == 0 \text{ to } x.size(0), j &== 0 \text{ to } y.size(0), y[j] \geq 0, \text{ and } i \neq y[j] \text{ for all } i \text{ and } j.
\end{aligned}
$$
式中 $y[j]$ 表示的标签，而$x\Big[ y[j] \Big]$ 的意思就是“标签所在神经元（即对应类别的预测值）”，又因为要求 $i \neq y[j]$，所以 $x[i]$ 表示的就是“非标签所在神经元（即非对应类别的预测值）”，当 $x \Big[ y[j] \Big] - x[i] < 0$ 时，$\text{max} \Bigg( 0, 1 - \Big(x \Big[ y[j] \Big] - x[i] \Big) \Bigg)$ 定会 $> 1$，而梯度下降是要朝着 loss 减小的方向前进的，所以为了让 loss 更小，则必然会希望 $x \Big[ y[j] \Big] - x[i] > 0$，即希望 $x \Big[ y[j] \Big] - x[i]$ 朝着 $> 0$ 的方向前进。说了那么多，其实用一句话总结就是 loss 希望非标签所在神经元要大于非标签所在神经元且要差值大于1.


**举个栗子**: 四分类任务, 样本x属于0类和3类，此时样本 x 的标签是 [0, 3, -1, -1]（注意不是不是[1, 0, 0, 1]），详情先看看下面的代码吧：
```python
# ---------------------------------------------- 11 Multi Label Margin Loss -----------------------------------------
# flag = 0
flag = 1
if flag:

    x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
    y = torch.tensor([[0, 3, -1, -1]], dtype=torch.long)  # 这个Tensor数据类型一定要是long

    loss_f = nn.MultiLabelMarginLoss(reduction='none')

    loss = loss_f(x, y)

    print(loss)


# output:
tensor([0.8500])
```

下面来分析下计算过程.

对于下面这样的两个中张量:
```python
x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
y = torch.tensor([[0, 3, -1, -1]], dtype=torch.long)  # 这个标签Tensor的数据类型一定要是long
```
（上面的 x 和 y 是用二维tensor表示的，y.shape = x.shape = (1, 4)，这是因为第一个维度表示的 batch_size，然而下面为了叙述方便就不考虑第1个维度了）\
其中 x 对应预测值，y 对应对应标签，$y[j] \geq 0$ 的表示对应标签，$y[j] \lt 0$ 的表示非对应标签。\
为了叙述方便，这里规定用下标 $j$ 索引标签 $y$，用下标 $i$ 索引预测值 $x$. 按照上述计算公式的要求， \
因为 $\text{x.size(0) = 4, y.size(0) = 4}$，所以有 $i \in [0, 4]$，$j \in [0, 4]$， \
又因为 $y[j] \geq 0$，所以 $j$ 只能 $\in \{ 0, 1 \}$，因而 $y[j] \in \{ 0, 3 \}$， \
又因为 $i \neq y[j]$，所以 $i$ 只能 $\in \{ 1, 2 \}$，因而 $x[i] \in \{ 0.2, 0.4 \}$，而 $x[ y[j] ] \in \{ 0.1, 0.8 \}$

将上述数值带入计算公式，便有：\
当 $j=0$ 时，$y[j] = 0, x[y[j]] = x[0]$，带入公式得：(这里为了叙述方便，暂时不考虑分母)
$$
\text{item}_1 = \sum_{i \in \{1, 2 \}, \; j=0} = \text{max(0, 1-(x[0] - x[1])) + max(0, 1-(x[0] - x[2]))}
$$

$$
\text{item}_2 = \sum_{i \in \{1, 2 \}, \; j=1} = \text{max(0, 1-(x[3] - x[1])) + max(0, 1-(x[3] - x[2]))}
$$

将上面两式汇总，并带入分母，便有得到最后的 loss：
$$
\text{loss}(x, y) = \sum_{ij} = \frac{\text{item}_1 + \text{item}_2}{4}
$$


下面是手动计算的代码：
```python
# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:
    x = x[0]
    item_1 = (1-(x[0] - x[1])) + (1 - (x[0] - x[2]))    # [0]
    item_2 = (1-(x[3] - x[1])) + (1 - (x[3] - x[2]))    # [3]

    loss_h = (item_1 + item_2) / x.shape[0]

    print(loss_h)


# output:
tensor(0.8500)
```


## nn.SoftMarginLoss
```python
class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and target tensor :math:`y`
    (containing 1 or -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    Args:
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
        - Input: :math:`(*)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftMarginLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.soft_margin_loss(input, target, reduction=self.reduction)
```
**功能**: 计算二分类的 logistic 损失。
主要参数:
- **reduction**: 计算模式, 可为none/sum/mean

**计算公式**：
$$
\text{loss}(x, y) = \sum_i \frac{log(1 + exp(-y[i] * x[i]))}{x.nelement()}
$$
上式中，标签 $y[i]$ 只能取 $\{ -1, 1 \}$ 中的元素（注意不是 $\{ 0, 1 \}$）

nn.SoftMarginLoss 是逐元素计算 loss


代码示例：
```python
# ---------------------------------------------- 12 SoftMargin Loss -----------------------------------------
# flag = 0
flag = 1
if flag:

    inputs = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
    target = torch.tensor([[-1, 1], [1, -1]], dtype=torch.float)

    loss_f = nn.SoftMarginLoss(reduction='none')

    loss = loss_f(inputs, target)

    print("SoftMargin: ", loss)

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    idx = 0

    inputs_i = inputs[idx, idx]
    target_i = target[idx, idx]

    loss_h = np.log(1 + np.exp(-target_i * inputs_i))

    print(loss_h)


# output:
SoftMargin:  
tensor([[0.8544, 0.4032],
        [0.4741, 0.9741]])

# 手动计算结果：
tensor(0.8544)

```

## nn.MultiLabelSoftMarginLoss
```python
class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input :math:`x` and target :math:`y` of size
    :math:`(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
                         + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.nElement}() - 1\right\}`,
    :math:`y[i] \in \left\{0, \; 1\right\}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
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
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, label targets padded by -1 ensuring same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """
    __constants__ = ['weight', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(MultiLabelSoftMarginLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, weight=self.weight, reduction=self.reduction)
```
**功能**: 是 nn.SoftMarginLoss 的多标签版本

主要参数:
- **weight**: 各类别的loss设置权值
- **reduction**: 计算模式, 可为none/sum/mean

**计算公式**:
$$
loss(x, y) = -\frac{1}{C} * \sum_i y[i] * log \Big(\frac{1}{1 + exp(-x[i])} \Big) + (1 - y[i]) * log \Big(\frac{exp(-x[i])}{1 + exp(-x[i])} \Big)
$$
式中，$C$ 表示类别数量，$y$ 表示真实标签，$x$ 表示预测值. 这里 $y \in {0, 1}$.

举个例子:
对于一个具有3个标签的分类任务来说，样本 x 属于这3个标签中的其中2个标签，则样本 x 的标签为 [0, 1, 1].


代码示例：
```python
# ---------------------------------------------- 13 MultiLabel SoftMargin Loss -----------------------------------------
# flag = 0
flag = 1
if flag:

    inputs = torch.tensor([[0.3, 0.7, 0.8]])
    target = torch.tensor([[0, 1, 1]], dtype=torch.float)

    loss_f = nn.MultiLabelSoftMarginLoss(reduction='none')

    loss = loss_f(inputs, target)

    print("MultiLabel SoftMargin: ", loss)

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    i_0 = torch.log(torch.exp(-inputs[0, 0]) / (1 + torch.exp(-inputs[0, 0])))

    i_1 = torch.log(1 / (1 + torch.exp(-inputs[0, 1])))
    i_2 = torch.log(1 / (1 + torch.exp(-inputs[0, 2])))

    loss_h = (i_0 + i_1 + i_2) / -3

    print(loss_h)


# output:
MultiLabel SoftMargin:  
tensor([0.5429])

# 手动计算结果：
tensor(0.5429)
```

## nn.MultiMarginLoss
```python
class MultiMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
    output :math:`y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)-1`):

    For each mini-batch sample, the loss in terms of the 1D input :math:`x` and scalar
    output :math:`y` is:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`
    and :math:`i \neq y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D :attr:`weight` tensor into the constructor.

    The loss function then becomes:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p)}{\text{x.size}(0)}

    Args:
        p (int, optional): Has a default value of :math:`1`. :math:`1` and :math:`2`
            are the only supported values.
        margin (float, optional): Has a default value of :math:`1`.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
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
    """
    __constants__ = ['p', 'margin', 'weight', 'reduction']

    def __init__(self, p=1, margin=1., weight=None, size_average=None,
                 reduce=None, reduction='mean'):
        super(MultiMarginLoss, self).__init__(weight, size_average, reduce, reduction)
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin

    def forward(self, input, target):
        return F.multi_margin_loss(input, target, p=self.p, margin=self.margin,
                                   weight=self.weight, reduction=self.reduction)
```
**功能**: 计算多分类的折页损失

主要参数:
- **p**: 可选1或2
- **weight**: 各类别的loss设置权值
- **margin**: 边界值
- **reduction**: 计算模式, 可为none/sum/mean


计算公式：
$$
\text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}
$$

$$
\begin{aligned}
    where & \quad x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}, 
y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}, 
0 \leq y[j] \leq \text{x.size}[0] - 1, \\
and & \quad i \neq y[j] \quad \text{for} \; \text{all} \; i \; \text{and} \; j.
\end{aligned}
$$
y 表示标签所在的神经元，x[y] 表示标签所在神经元的输出值，x[i] 表示非标签所在神经元的输出值.（这一点和 nn.MultiLabelMarginLoss 有些类似.）


代码示例：
```python
# ---------------------------------------------- 14 Multi Margin Loss -----------------------------------------
# flag = 0
flag = 1
if flag:

    x = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3]])
    y = torch.tensor([1, 2], dtype=torch.long)

    loss_f = nn.MultiMarginLoss(reduction='none')

    loss = loss_f(x, y)

    print("Multi Margin Loss: ", loss)

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    x = x[0]
    margin = 1

    i_0 = margin - (x[1] - x[0])
    # i_1 = margin - (x[1] - x[1])
    i_2 = margin - (x[1] - x[2])

    loss_h = (i_0 + i_2) / x.shape[0]

    print(loss_h)


# output:
Multi Margin Loss:  
tensor([0.8000, 0.7000])

# 手动计算结果：
tensor(0.8000)
```


## nn.TripletMarginLoss
```python
class TripletMarginLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        margin (float, optional): Default: :math:`1`.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
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
        - Input: :math:`(N, D)` where :math:`D` is the vector dimension.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']

    def __init__(self, margin=1.0, p=2., eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction='mean'):
        super(TripletMarginLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)
```
**功能**: 计算三元组损失, 人脸验证中常用

主要参数:
- **p**:范数的阶, 默认为2
- **margin**: 边界值
- **reduction**: 计算模式, 可为none/sum/mean

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_loss6.jpg" width = 80% height = 80% />
</div>

计算公式：
$$
L(a, p, n) = \text{max} {d(a_i, p_i) - d(a_i, n_i) + margin, 0}
$$

$$
d(x_i, y_i) = \Vert x_i - y_i \Vert _p
$$


代码示例：
```python
# ---------------------------------------------- 15 Triplet Margin Loss -----------------------------------------
# flag = 0
flag = 1
if flag:

    anchor = torch.tensor([[1.]])
    pos = torch.tensor([[2.]])
    neg = torch.tensor([[0.5]])

    loss_f = nn.TripletMarginLoss(margin=1.0, p=1)

    loss = loss_f(anchor, pos, neg)

    print("Triplet Margin Loss", loss)

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:

    margin = 1
    a, p, n = anchor[0], pos[0], neg[0]

    d_ap = torch.abs(a-p)
    d_an = torch.abs(a-n)

    loss = d_ap - d_an + margin

    print(loss)



# output:
Triplet Margin Loss 
tensor(1.5000)

# 手动计算结果
tensor([1.5000])
```


## nn.HingeEmbeddingLoss
```python
class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`
    (containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as :math:`x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float, optional): Has a default value of `1`.
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
        - Input: :math:`(*)` where :math:`*` means, any number of dimensions. The sum operation
          operates over all the elements.
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=1.0, size_average=None, reduce=None, reduction='mean'):
        super(HingeEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, margin=self.margin, reduction=self.reduction)
```
**功能**: 计算两个输入的相似性,常用于非线性 embedding 和半监督学习

主要参数:
- **margin**: 边界值
- **reduction**: 计算模式, 可为none/sum/mean

计算公式：
$$l_n = 
\begin{cases}
    x_n, \qquad & \text{if} \; y_n = 1, \\
    max{0, \Delta - x_n}, \qquad & \text{if} \; y_n = -1,
\end{cases}
$$

这里的 $\Delta$ 也相当于是 margin，默认是1. \
**特别注意**: 输入x应为两个输入之差的绝对值


代码示例：
```python
# ---------------------------------------------- 16 Hinge Embedding Loss -----------------------------------------
# flag = 0
flag = 1
if flag:

    inputs = torch.tensor([[1., 0.8, 0.5]])
    target = torch.tensor([[1, 1, -1]])

    loss_f = nn.HingeEmbeddingLoss(margin=1, reduction='none')

    loss = loss_f(inputs, target)

    print("Hinge Embedding Loss", loss)

# --------------------------------- compute by hand
# flag = 0
flag = 1
if flag:
    margin = 1.
    loss = max(0, margin - inputs.numpy()[0, 2])

    print(loss)



# output:
Hinge Embedding Loss 
tensor([[1.0000, 0.8000, 0.5000]])

# 手动计算结果：
0.5
```


## nn.CosineEmbeddingLoss
```python
class CosineEmbeddingLoss(_Loss):
    r"""Creates a criterion that measures the loss given input tensors
    :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    The loss function for each sample is:

    .. math::
        \text{loss}(x, y) =
        \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y = 1 \\
        \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
        \end{cases}

    Args:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
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
    """
    __constants__ = ['margin', 'reduction']

    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)
```
**功能**: 采用余弦相似度计算两个输入的相似性，也是常用于非线性 embedding 和半监督学习。

计算公式：
$$
\text{loss}(x, y) =
        \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y = 1 \\
        \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
        \end{cases}
$$

$$
cos(\theta) = \frac{A \cdot B}{\Vert A \Vert \Vert B \Vert} = \frac{\sum^n_{i=1} A_i \times B_i}{\sqrt{\sum^n_{i=1}(A_i)^2} \times \sqrt{\sum^n_{i=1}(B_i)^2}}.
$$

主要参数:
- **margin**: 可取值[-1, 1]（因为cos的值域就是[-1,1]）, 推荐为[0, 0.5]
- **reduction**: 计算模式, 可为none/sum/mean


## nn.CTCLoss
```python
class CTCLoss(_Loss):
    r"""The Connectionist Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

    Args:
        blank (int, optional): blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Shape:
        - Log_probs: Tensor of size :math:`(T, N, C)`,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
          The logarithmized probabilities of the outputs (e.g. obtained with
          :func:`torch.nn.functional.log_softmax`).
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target\_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N)`, where :math:`N = \text{batch size}`.

    Example::

        >>> T = 50      # Input sequence length
        >>> C = 20      # Number of classes (including blank)
        >>> N = 16      # Batch size
        >>> S = 30      # Target sequence length of longest target in batch
        >>> S_min = 10  # Minimum target length, for demonstration purposes
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        >>>
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    .. Note::
        In order to use CuDNN, the following must be satisfied: :attr:`targets` must be
        in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
        :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
        dtype :attr:`torch.int32`.

        The regular implementation uses the (more common in PyTorch) `torch.long` dtype.


    .. include:: cudnn_deterministic.rst

    """
    __constants__ = ['blank', 'reduction']

    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super(CTCLoss, self).__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction,
                          self.zero_infinity)
```
**功能**: 计算 CTC 损失, 解决时序类数据的分类。比如OCR中输入和输出长度不确定的情况下我们怎么去计算他的分类。

英文全称：Connectionist Temporal Classification

参考文献：
《A. Graves et al.: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks》

主要参数:
- **blank**: blank label
- **zero_infinity**: 无穷大的值或梯度置0
- **reduction**: 计算模式, 可为none/sum/mean


代码示例：
```python
# ---------------------------------------------- 18 CTC Loss -----------------------------------------
# flag = 0
flag = 1
if flag:
    T = 50      # Input sequence length
    C = 20      # Number of classes (including blank)
    N = 16      # Batch size
    S = 30      # Target sequence length of longest target in batch
    S_min = 10  # Minimum target length, for demonstration purposes

    # Initialize random batch of input vectors, for *size = (T,N,C)
    inputs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(inputs, target, input_lengths, target_lengths)

    print("CTC loss: ", loss)


# output:
CTC loss:  
tensor(7.5385, grad_fn=<MeanBackward0>)
```

# 总结：PyTorch 中的18中损失函数

1. nn.CrossEntropyLoss
2. nn.NLLLoss
3. nn.BCELoss
4. nn.BCEWithLogitsLoss
5. nn.L1Loss
6. nn.MSELoss
7. nn.SmoothL1Loss
8. nn.PoissonNLLLoss
9. nn.KLDivLoss
10. nn.MarginRankingLoss
11. nn.MultiLabelMarginLoss
12. nn.SoftMarginLoss
13. nn.MultiLabelSoftMarginLoss
14. nn.MultiMarginLoss
15. nn.TripletMarginLoss
16. nn.HingeEmbeddingLoss
17. nn.CosineEmbeddingLoss
18. nn.CTCLoss


# 参考文献
[1] DeepShare.net > PyTorch框架
