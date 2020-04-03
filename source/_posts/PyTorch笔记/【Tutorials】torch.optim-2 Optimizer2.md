---
title: 
date: 2020-02-16
tags:
categories: ["PyTorch笔记"]
mathjax: true
---



# leaning rate 学习率
在之前已经讲到过梯度下降的更新策略如下式所示：
$$w_{i+1} = w_i - \text{grad}(w_i)$$

下面以 $y = f(x) = 4x^2$ 为例来观察梯度下降公式的使用，以及该公式可能存在的问。\
其中 x 可类比为参数，y 可类比为 loss \
而 y 的导数就是：$y' = f'(x) = 8x$ \

使用梯度下降更新策略更新参数的过程：\
$x_0 = 2, y_0 = 16, f'(x_0) = 16$ \
$x_1 = x_0 - f'(x_0) = 2 - 16 = -14, y_1 = 784, f'(x_1) = -112$ \
$x_2 = x_1 - f'(x_1) = -14 + 112 = 98, y_2 = 38416, f'(x_2) = 784$ \
......

可以看到 y 值(loss)越来越大，参数 x 的尺度也越来越大。


方程 $y = f(x) = 4x^2$ 曲线绘制代码
```python
# -*- coding:utf-8 -*-
"""
@file name  : learning_rate.py
# @author     : TingsongYu https://github.com/TingsongYu
@date       : 2019-10-16
@brief      : 梯度下降的学习率演示
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)


def func(x_t):
    """
    y = (2x)^2 = 4*x^2      dy/dx = 8x
    """
    return torch.pow(2*x_t, 2)


# init
x = torch.tensor([2.], requires_grad=True)


# ------------------------------ plot data ------------------------------
flag = 0
# flag = 1
if flag:

    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.plot(x_t.numpy(), y.numpy(), label="y = 4*x^2")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
```
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_lr1.jpg" width = 60% height = 60% />
</div>


## 学习率对梯度下降的影响
```python
# ------------------------------ gradient descent ------------------------------
# flag = 0
flag = 1
if flag:
    iter_rec, loss_rec, x_rec = list(), list(), list()

    lr = 0.2    # /1. /.5 /.2 /.1 /.125
    max_iteration = 4   # /1. 4     /.5 4   /.2 20 200

    for i in range(max_iteration):

        y = func(x)
        y.backward()

        print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(
            i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))

        x_rec.append(x.item())

        x.data.sub_(lr * x.grad)    # x -= x.grad  数学表达式意义:  x = x - x.grad    # 0.5 0.2 0.1 0.125
        x.grad.zero_()

        iter_rec.append(i)
        loss_rec.append(y)

    plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")

    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.subplot(122).plot(x_t.numpy(), y.numpy(), label="y = 4*x^2")
    plt.grid()
    y_rec = [func(torch.tensor(i)).item() for i in x_rec]
    plt.subplot(122).plot(x_rec, y_rec, '-ro')
    plt.legend()
    plt.show()
```
<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_lr2.jpg" width = 70% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_lr3.jpg" width = 70% height = 50% />
                </div>
            </td>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_lr4.jpg" width = 70% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图1 &nbsp; 学习率对梯度下降的影响(左边学习率为1，中间学习率为0.2,右边学习率为0.125)</center>
</html>

观察上图可知，当学习率为0.125时，y 通过迭代一步就可以到达底部，这是为什么呢？\
因为，我们已经知道了 y 的表达式 $y = f(x) = 4x^2$，我们希望 y（loss）达到最小值，\
y 达到最小值时的 x（参数）取值是 0，也就是我说我们已经知道了梯度下降的结果是 0.
$$
\begin{aligned}
    w_{i+1} = w_i - lr * \text{grad}(w_i) \Rightarrow x_1 &= x_0 - lr * \text{grad}(x_0) \\
                                                   0 &= 2 - lr * \text{grad}(2) \\
                                                   0 &= 2 - lr * 16 \\
                                      \Rightarrow lr &= \frac{1}{8} = 0.125
\end{aligned}
$$
也就是说我们这里事先知道了 y 的表达式，然后求取 y 的极小值点所对应的参数 $x^\*$，然后将 $x^\*$ 带入梯度下降公式，自然就能求出最直接的学习率。当然现实应用中一般都是不知道表达式的。

# momentum 动量
**Momentum(动量，冲量)**：结合当前梯度与上一次更新信息，用于当前更新。

**指数加权平均**：(在时间序列分析中常用)
$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \theta_{t}
$$

$\theta_t$ 表示当前时刻的参数(权重) \
$v_{t-1}$ 表示上一时刻参数的指数加权平均值 \
$v_t$ 表示当前时刻参数的指数加权平均值 \

$$
\begin{aligned}
    v_{100} &= \beta \cdot v_{99} + (1 - \beta) \cdot \theta_{100} \\
            &= (1 - \beta) \cdot \theta_{100} + \beta \cdot (\beta \cdot v_{98} + (1 - \beta) \cdot \theta_{99}) \\
            &= \sum^N_i (1 - \beta) \cdot \beta^i \cdot \theta_{N - i}
\end{aligned}
$$

观察上面的公式可知，当前时刻参数 $v_{100}$ 等于之前历史时刻参数的加权平均，距离 $v_{100}$ 越远的参数所占的权重越小(呈指数下降趋势)，而当前时刻的参数 $v_{100}$ 所占的权重最大。


探索超参数 $\beta$ 设置为不同值时对加权平均的影响：



# torch.optim.SGD


# Pytorch 的十种优化器



# 参考文献
[1] DeepShare.net > PyTorch框架
