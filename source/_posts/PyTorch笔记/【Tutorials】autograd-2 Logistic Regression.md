---
title: 
date: 2019-09-11
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述逻辑回归模型以及逻辑回归和线性回归之间的关系，并利用上一节中所讲的autograd来做一个逻辑回归算法的示例。
<!-- mored -->


逻辑回归是**线性**的**二分类**模型模型表达式:
$$
\begin{aligned}
    y &= f(WX + b)  \\
    f(x) &= \frac{1}{1 + e^{-x}}  \tag{1}
\end{aligned}
$$

$$
class = 
\begin{dcases}
    0, \qquad y < 0.5  \\
    1, \qquad y \ge 0.5
\end{dcases}
$$

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/autograd_sigmoid.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp; Sigmoid曲线</center>

因变量 $y$ 是自变量 $x$ 的线性组合 $WX +b$ 再输入到 $f(x)$ 函数中，$f(x)$ 称为Sigmoid函数，也称为Logistic函数。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/autograd_logistic_regression.jpg" width = 60% height = 60% />
</div>
<center>图2 &nbsp;逻辑回归与线性回归的关系</center>

&emsp; 逻辑回归可以理解为在线性回归的基础上加了一个激活函数（Sigmoid函数），如果将Sigmoid函数舍弃，单纯留下 $WX + b$，则逻辑回归模型还是能够做二分类的，这时候当 $WX + b > 0时$（等价于逻辑回归中的 $y>0.5$）就判别为类别1，当 $WX + b < 0$ 时（等价于逻辑回归中的 $y<0.5$），就判别为类别0. 但是为了更好的利用分类置信度，所以要将线性回归模型再输入到 Sigmoid 函数中。


**逻辑回归也称对数几率回归**，对数几率回归如公式(2)所示，

$$ln\frac{y}{1 - y} = WX + b  \tag{2}$$
几率就是概率，式(2)中的 $\frac{y}{1 - y}$ 可以理解成判别为正样本的概率，再对 $\frac{y}{1 - y}$ 取对数就得到了**对数几率**.

公式(2) 和公式(1) 是等价的，推导过程如下：
$$
\begin{aligned}
    ln \frac{y}{1 - y} &= WX + b  \\
    \frac{y}{1 - y} &= e^{WX + b}  \\
    y &= e^{WX + b} - y * e^{WX + b}  \\
    y(1 + e^{WX + b}) &= e^{WX + b}  \\
    y &= \frac{e^{WX + b}}{1 + e^{WX + b}} = \frac{1}{1 + e^{-(WX + b)}}
\end{aligned}
$$

比较公式(2)和线性回归模型 $y = WX + b$ 会发现，线性回归是用 $WX + b$ 去拟合 $y$，而对数几率回归是用 $WX + b$ 去拟合对数几率 $ln \frac{y}{1 - y}$，所以这也是对数几率回归名字的由来。


# 逻辑回归代码示例

&emsp; 机器学习模型训练的一般步骤：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/autograd_ml_train_step.jpg" width = 60% height = 60% />
</div>
<center>图3 &nbsp;机器学习模型训练步骤</center>

step1: 获取数据，可能会涉及数据采集、划分、清洗、预处理；\
step2: 选择模型; \
step3: 根据不同的任务选择不同的损失函数，例如在线性回归任务中可以选择均方差(MSE)损失函数，在分类任务中可以选择交叉熵损失函数；\
step4: 有了loss后，就可以求取梯度，得到梯度后就可以采取某一种优化器来优化即将要训练的模型；\
step5: 有了上述配置之后，就可以开始迭代训练了。


```python
# -*- coding: utf-8 -*-
"""
# @file name  : lesson-05-Logsitic-Regression.py
# @author     : tingsongyu
# @date       : 2019-09-03 10:08:00
# @brief      : 逻辑回归模型训练
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)


# ============================ step 1/5 生成数据 ============================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2)
y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100, 1)
x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2)
y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100, 1)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)


# ============================ step 2/5 选择模型 ============================
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()   # 实例化逻辑回归模型


# ============================ step 3/5 选择损失函数 ============================
loss_fn = nn.BCELoss()

# ============================ step 4/5 选择优化器   ============================
lr = 0.01  # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# ============================ step 5/5 模型训练 ============================
for iteration in range(1000):

    # 前向传播
    y_pred = lr_net(train_x)

    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), train_y)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 清空梯度
    optimizer.zero_grad()

    # 绘图
    if iteration % 20 == 0:

        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break
```

