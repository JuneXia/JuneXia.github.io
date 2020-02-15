---
title: 【深度学习笔记 paper】Center Loss
date: 2019-04-14 17:28:05
tags:
categories: ["深度学习笔记"]
mathjax: true
---


&emsp; 在使用CNN做分类任务时，我们通常会用softmax函数来计算网络的损失，softmax函数公式如下：

$$
L_S = -\sum^{m}_{i=1}log\frac{e^{\boldsymbol{W}_{y_i}^T x_i + b_{y_i}}}{\sum^n_{j=1} e^{\boldsymbol{W}_j^T x_i + b_j}}
$$

而softmax只能将不同的类别分隔开（如图1 Separable Features），但每个类别相互之间并不具有很好的区分度（如图1 Discriminative Features）。

[图1 分类任务和人脸识别任务对Features的要求比较](../../images/ml/center-loss1.png)


鉴于此，Y. Wen等人提出了 CenterLoss 来解决这个问题，其核心思想是希望类内尽量紧，
类间尽量开，其计算公式如下：

[](../../images/ml/center-loss2.png)

由公式可以看出CenterLoss只是在softmax函数的基础上增加了一个Lc，作者认为这种联合监督训练既可以有效做分类任务，也可以使得类与类之间具有更好的辨识力（Discriminative)。

## 我的理解
&emsp; 公式 Lc 中的 $\boldsymbol{c}_{y_i}$ 表示第 $y_i$ 个类别的中心。

好的，那我先假设网络输出特征 $\boldsymbol{F}_{1 \times n}$，神经网路最后一个分类层权重为 $\boldsymbol{W}_{n \times m}$，实际上 $\boldsymbol{W}_{n \times m}$ 的每一列向量都是该类别的中心。
但按作者论文中的实现思想，作者并没有使用这个 $\boldsymbol{W}$ 来作为 $\boldsymbol{c}_{y_i}$，而是重新定义了一个 $\boldsymbol{w}$ 来进行计算来作为 $\boldsymbol{c}_{y_i}$（当然这个 $\boldsymbol{w}$ 也是 $n \times m$ 的矩阵），且 $\boldsymbol{w}$ 并没有使用反向传播更新，而是单独更新的。


## 实际应用
在人脸识别中实测有改进，但改进仍然不大，都是千分位的提升。