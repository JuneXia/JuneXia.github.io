---
title: 
date: 2020-04-26
tags:
categories: ["深度学习笔记"]
mathjax: true
---

# 感受野的理解与计算

**什么是感受野**
在卷积神经网络中,决定某一层输出结果中一个元素所对应的输入层的区域的大小,被称作感受野(receptive field),通俗的解释是,输出feature map上的一个单元对应输入层上的区域大小。
<!-- more -->


<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/receptive-field1.jpg" width = 70% height = 70% />
</div>
图片来自文献[1]

**感受野计算公式**：
$$
F_i = (F_{i+1} - 1) \times Stride + Ksize
$$


举个例子吧，为了表述方便，我们命名上图中的input为$F_1$，经过conv1后得到的feature-map记作 $F_2$，经过 pooling 后的 feature-map 记作 $F_3$ . 
如图所示，pooling 后的 feature-map $F_3$ 有 $2 \times 2$ 个像素，每个像素在 $F_2$ 层对应的感受野是：
$$
\begin{aligned}
    F_2 &= (F_{3} - 1) \times 2 + 2
        &= (1 - 1) \times 2 + 2
        &= 2
\end{aligned}
$$
这里 pooling 的stride=2，Ksize=2.

所以 $F_3$ 的每个像素在 $F_2$ 中对应的感受野是 $2 \times 2$.

$F_3$ 的每个像素在 $F_1$ 中对应的感受野是：
$$
\begin{aligned}
    F_1 &= (F_{2} - 1) \times 2 + 3
        &= (2 - 1) \times 2 + 3
        &= 5
\end{aligned}
$$
这里 conv1 的stride=2，Ksize=3.

所以 $F_3$ 的每个像素在 $F_1$ 中对应的感受野是 $5 \times 5$.

----------------

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/receptive-field2.jpg" width = 70% height = 70% />
</div>


网络中的亮点:

&emsp; VGG论文中提到,通过堆叠多个3x3的卷积核来替代大尺度卷积核(减少所需参数)。可以通过堆叠两个3x3的卷积核替代5x5的卷积核,堆叠三个3x3的卷积核替代7x7的卷积核(拥有相同的感受野)。
使用7x7卷积核所需参数,与堆叠三个3x3卷积核所需参数(假设输入输出channel为C)如下：

$7 \times 7 \times C \times C = 49 C^2$

$
3  \times 3 \times C \times C + 3 \times 3 \times C \times C + 3 \times 3 \times C \times C = 27 C^2
$


# 参考文献
[1] [4.1 VGG网络详解及感受野的计算](https://www.bilibili.com/video/BV1q7411T7Y6?from=search&seid=3037291823851065458)