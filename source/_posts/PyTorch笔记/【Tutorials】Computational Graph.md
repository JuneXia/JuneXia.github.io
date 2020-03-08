---
title: 
date: 2019-09-11
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述计算图的概念以及tensor的一些属性(is_leaf、grad_fn等)，这些属性在计算图中是非常重要的。
<!-- more -->

计算图(Computational Graph)是用来描述运算的有向无环图。
计算图主要有两个元素：节点(Node)和边(Edge)，节点表示数据，如向量、矩阵、张量，边表示运算，如加减乘除卷积等。

# 计算图与梯度求导
用计算图表示：$y = (x + w) * (w + 1)$

可以将上式拆分成：
$$
\begin{aligned}
    a &= x + w \\
    b &= w + 1 \\
    y &= a * b \\
\end{aligned}
$$
则：
$$
\begin{aligned}
    \frac{\partial y}{\partial w} &= 
    \frac{\partial y}{\partial a} \frac{\partial a}{\partial w} + 
    \frac{\partial y}{\partial b} \frac{\partial b}{\partial w} \\

    &= b * 1 + a * 1 \\
    &= b + a \\
    &= (w + 1) + (x + w) \\
    &= 2*w + x + 1 \\
    &= 2*1 + 2 + 1 = 5
\end{aligned}
$$

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/computational_graph.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp;  计算图</center>

y 对 w 求导，在计算图中实际上就是要找到所有 y 到 w 的路径，把路径上的导数进行求和就得到了 y 对 w 的导数。


> 这里补充讲一些张量的属性，这对理解计算图是很重要的。
> <div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/computational_graph2.jpg" width = 60% height = 60% /></div><center>图2 &nbsp;  张量的属性</center>
> 1. tensor属性：is_leaf，用于指示张量是叶子节点或非叶子节点
> 
> &emsp; **叶子节点**：用户创建的节点称为叶子节点，如图1中的x和w. 叶子节点是整个计算图的根基，在前向传播中，后面layer的节点都要依赖于前面的叶子节点，而在反向传播中，所有梯度的计算都要依赖于叶子节点。\
> 设置叶子节点和非叶子节点这些概念是因为在反向传播的过程中，非叶子节点的梯度内存会被释放，以节约内存。但是如果想要保存某个非叶子节点的梯度该怎么办呢，pytorch中为tensor提供了一个 retain_grad() 的方法来保存张量的梯度。
> 
> 2. tensor属性：grad_fn，用于记录创建该tensor时所用的方法(函数)
> 
> &emsp; w 和 x 是用户创建张量，它没有通过任何function来生成，所以它们的grad_fn都是None.\
> 张量的这一属性会在梯度求导的过程中(即反向传播的时候)被用到，如果某个节点是通过乘法得到的，那么该节点的grad_fn应该是<MulBackward>，当该节点在反向求导的过程中就会调用相应乘法求导法则。同理张量grad_fn为<AddBackward>时情况类似。


下面通过代码来看看这些性质。
```python
# -*- coding:utf-8 -*-
"""
@file name  : lesson-04-Computational-Graph.py
@author     : tingsongyu
@date       : 2018-08-28
@brief      : 计算图示例
"""
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
a.retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)



# output:
tensor([5.])
is_leaf:
 True True False False False

gradient:
 tensor([5.]) 
 tensor([2.]) 
 tensor([2.]) # a是非也自己点，但是可以通过a的retain_grad()函数来保存梯度。
 None 
 None

grad_fn:
 None 
 None 
 <AddBackward0 object at 0x7f47259e0160> 
 <AddBackward0 object at 0x7f47259e0198> 
 <MulBackward0 object at 0x7f47259e0128>
```


# 动态图与静态图

典型的代表便是PyTorch与TensorFlow，... ，没什么好讲的。。。



# 参考文献
[1] DeepShare.net > PyTorch框架