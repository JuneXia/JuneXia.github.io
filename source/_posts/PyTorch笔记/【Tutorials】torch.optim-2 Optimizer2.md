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

不同学习率对梯度下降更新速度的影响：
```python
# ------------------------------ multi learning rate ------------------------------

# flag = 0
flag = 1
if flag:
    iteration = 100
    num_lr = 10
    lr_min, lr_max = 0.01, 0.2  # .5 .3 .2

    lr_list = np.linspace(lr_min, lr_max, num=num_lr).tolist()
    loss_rec = [[] for l in range(len(lr_list))]
    iter_rec = list()

    for i, lr in enumerate(lr_list):
        x = torch.tensor([2.], requires_grad=True)
        for iter in range(iteration):

            y = func(x)
            y.backward()
            x.data.sub_(lr * x.grad)  # x.data -= x.grad
            x.grad.zero_()

            loss_rec[i].append(y.item())

    for i, loss_r in enumerate(loss_rec):
        plt.plot(range(len(loss_r)), loss_r, label="LR: {}".format(lr_list[i]))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')
    plt.show()
```
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_lr5.jpg" width = 60% height = 60% />
</div>


# momentum 动量
**Momentum(动量，冲量)**：结合当前梯度与上一次更新信息，用于当前更新。

## 指数加权平均

在讲momentum之前，先来讲一下指数加权平均的概念。

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
            &= ... \\
            &= (1 - \beta) \cdot \theta_{100} + (1 - \beta) \cdot \beta \cdot \theta_{99} + (1 - \beta) \cdot \beta^2 \cdot \theta_{98} + \beta^3 \cdot v_{97} \\
            &= \sum^N_i (1 - \beta) \cdot \beta^i \cdot \theta_{N - i}
\end{aligned}
$$

观察上面的公式可知，当前时刻参数 $v_{100}$ 等于之前历史时刻参数的加权平均，距离 $v_{100}$ 越远的参数所占的权重越小(呈指数下降趋势)，而当前时刻的参数 $v_{100}$ 所占的权重最大。


探索超参数 $\beta$ 设置为不同值时对加权平均的影响：
```python
# -*- coding:utf-8 -*-
"""
@file name  : momentum.py
# @author     : TingsongYu https://github.com/TingsongYu
@date       : 2019-10-17
@brief      : 梯度下降的动量 momentum
"""
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
torch.manual_seed(1)


def exp_w_func(beta, time_list):
    return [(1 - beta) * np.power(beta, exp) for exp in time_list]


beta = 0.9
num_point = 100
time_list = np.arange(num_point).tolist()

# ------------------------------ exponential weight ------------------------------
# flag = 0
flag = 1
if flag:
    weights = exp_w_func(beta, time_list)

    plt.plot(time_list, weights, '-ro', label="Beta: {}\ny = B^t * (1-B)".format(beta))
    plt.xlabel("time")
    plt.ylabel("weight")
    plt.legend()
    plt.title("exponentially weighted average")
    plt.show()

    print(np.sum(weights))


# ------------------------------ multi weights ------------------------------
# flag = 0
flag = 1
if flag:
    beta_list = [0.98, 0.95, 0.9, 0.8]
    w_list = [exp_w_func(beta, time_list) for beta in beta_list]
    for i, w in enumerate(w_list):
        plt.plot(time_list, w, label="Beta: {}".format(beta_list[i]))
        plt.xlabel("time")
        plt.ylabel("weight")
    plt.legend()
    plt.show()
```
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_momentum1.jpg" width = 60% height = 60% />
</div>

观察上图可知，超参数 $\beta$ 的作用当于是**记忆周期**，当 $\beta$ 值越小，它的记忆周期越短（即只能记住较近时刻的作用），如上图中红色曲线所示。

$\beta$ 通常取 0.9，这表示相对过往所有历史时刻的参数，模型更关注最近的10次参数。

## PyTorch中的momentum

一般的梯度下降：
$$
w_{i + 1} = w_i - lr \cdot g(w_i)
$$

PyTorch中的梯度下降更新公式：(m对应指数加权平均中的$\beta$)
$$
\begin{aligned}
    v_i &= m \cdot v_{i-1} + g(w_i) \\
w_{i+1} &= w_i - lr \cdot v_i
\end{aligned}
$$
注意pytorch中的实现没有完全按照前面所说的指数加权平均来计算的，$g(w_i)$ 前面没有乘 $1 - m$

举个具体的例子吧：
$$
\begin{aligned}
    v_{100} &= m \cdot v_{99} + g(w_{100}) \\
            &= g(w_{100}) + m \cdot (m \cdot v_{98} + g(w_{99})) \\
            &= g(w_{100}) + m \cdot g(w_{99}) + m^2 \cdot v_{98} \\
            &= g(w_{100}) + m \cdot g(w_{99}) + m^2 \cdot g(w_{98}) + m^3 \cdot v_{97}
\end{aligned}
$$


代码示例：
```python
# ------------------------------ SGD momentum ------------------------------
# flag = 0
flag = 1
if flag:
    def func(x):
        return torch.pow(2*x, 2)    # y = (2x)^2 = 4*x^2        dy/dx = 8x

    iteration = 100
    m = 0.     # .9 .63

    lr_list = [0.01, 0.03]

    momentum_list = list()
    loss_rec = [[] for l in range(len(lr_list))]
    iter_rec = list()

    for i, lr in enumerate(lr_list):
        x = torch.tensor([2.], requires_grad=True)

        momentum = 0. if lr == 0.03 else m
        momentum_list.append(momentum)

        optimizer = optim.SGD([x], lr=lr, momentum=momentum)

        for iter in range(iteration):

            y = func(x)
            y.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_rec[i].append(y.item())

    for i, loss_r in enumerate(loss_rec):
        plt.plot(range(len(loss_r)), loss_r, label="LR: {} M:{}".format(lr_list[i], momentum_list[i]))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')
    plt.show()
```

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_momentum2.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_momentum3.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_momentum4.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
</html>

1. 上面左图，当不设置momentum时，较大的学习率相对小学习率会更快收敛；
2. 上面中图，为小学习率设置一个momentum（大学习率的momentum仍然为0），小学习率会更快抵达极值点，但随后有反弹并来回震荡，最后慢慢收敛；\
> 震荡是由于momentum太大所导致的，因为momentum太大，所以当loss抵达极小值时又受到之前梯度（惯性）的影响，从而还是以较大的步伐越过极小值点，这样就出现了来回震荡的现象。\
3. 上面右图，将momentum调小，此时小学习率也会很快的收敛。

> 而通过这个例子我们也可以知道选择合适的momentum是有助于loss更快达到极值点，但是开发中一般只有上帝视角才知道什么样的momentum值合适，所以实际开发中momentum一般还是设置为 0.9

# PyTorch中的优化器
## torch.optim.SGD
```python
class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        return loss
```
主要参数：
- **params**: 管理的参数组
- **lr**: 初始学习率
- **momentum**: 动量系数，$\beta$
- **weight_decay**: L2正则化系数
- **nesterov**: 是否采用NAG

NAG 参考文献：《On the importance of initialization and momentum in deep learning》


## Pytorch 的十种优化器

1. optim.SGD: 随机梯度下降法
2. optim.Adagrad: 自适应学习率梯度下降法
3. optim.RMSprop: Adagrad 的改进
4. optim.Adadelta: Adagrad 的改进
5. optim.Adam: RMSprop 结合 Momentum
6. optim.Adamax: Adam增加学习率上限
7. optim.SparseAdam: 稀疏版的梯度下降
8. optim.ASGD: 随机平均梯度下降
9. optim.Rprop: 弹性反向传播，通常在将全部数据加入一个batch去训练时会用到
10. optim.LBFGS: 对BFGS在内存消耗上的改进

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer_momentum5.jpg" width = 60% height = 60% />
</div>


# 参考文献
[1] DeepShare.net > PyTorch框架
