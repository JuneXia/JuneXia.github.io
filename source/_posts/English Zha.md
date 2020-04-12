---
title: 
date: 2018-06-25
tags:
categories: ["深度学习笔记"]
mathjax: true
---

In a convolutional fashion, we evaluate a small set (e.g. 4) **of** default boxes **of** different aspect ratios at each location in several feature maps with different scales. 
&emsp; &emsp; From `SSD:Single Shot MultiBox Detector`

> TIPS: 
> v. &ensp;n1 of n2 of n3  → v. &ensp;n1 n3 的 n2 \
> `in several feature maps with different scales. ` 在几个不同尺度的feature map中
> 
> Translate:
使用卷积的方法，我们在几个不同尺度的feature map中的每个位置  评估一小组(e.g. 4)不同长宽比的default boxes.

--------
<br>

To achieve high detection accuracy we **produce** predictions **of** different scales **from** feature maps **of** different scales, and explicitly separate predictions by aspect ratio. 
&emsp; &emsp; From `SSD:Single Shot MultiBox Detector`

> TIPS:
> v. &ensp;n1 of n2 from n3 of n4  →  n4 的 n3 &ensp;v. &ensp;n2 的 n1
> 
> Translate:
> 为了实现较高的检测精度，我们从不同尺度的特征图中生成不同尺度的预测，并通过纵横比明确地分离预测。

--------
<br>

Down sampling **is handled with** strided convolution **in** the depthwise convolutions **as well as in** the first layer.
&emsp; &emsp; From `MobileNets: Efficient Convolutional Neural Networks for Mobile Vision`

> TIPS: \
> n1 **be v.ed with** n2 **in** n3 **as well as in** n4. → n1 **在** n3 **和** n4 **中都 v.ed with** n2.
> 
> Translate:
> 下行采样在深度卷积和第一层卷积中都使用了strided convolution.

--------
<br>


When training MobileNets we **do not use** side heads or label smoothing **and** additionally reduce the amount image **of** distortions **by** limiting the size **of** small crops **that** are used in large Inception training. 
&emsp; &emsp; From `MobileNets: Efficient Convolutional Neural Networks for Mobile Vision`

> TIPS:
> additionally &ensp; adv. 此外；又，加之
> 
> Translate
> 当训练MobileNet的时候，我们不使用side heads或者标签平滑，并且通过限制small crops的尺寸来减少失真图片数量，这个small crops在大型 Inception 训练中会被用到。

--------
<br>

To summarize, we have highlighted two properties **that** are indicative of the requirement **that** the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space.

&emsp; &emsp; From `MobileNetV2: Inverted Residuals and Linear Bottlenecks`

> TIPS: \
> indicative(adj.象征的;指示的;表示…的) \
> be indicative of, 表明
> 
> Translate:\
> 综上所述，我们强调了两个性质，这两个性质表明了我们所关心的流形应该位于高维激活空间的低维子空间中

--------
<br>

We **note that** similar reports **where** non-linearity was helped were reported in [29] **where** non-linearity was removed from the input of the traditional residual block and that lead to improved performance on CIFAR dataset.

&emsp; &emsp; From `MobileNetV2: Inverted Residuals and Linear Bottlenecks`

> 蹩脚直译：\
> 我们注意到，在[29]中也有关于非线性层是有帮助的这样类似的报道，而non-linearity被从传统残差块中的输入中移除，从而提高了CIFAR数据集的性能。
> 
> 理解翻译：\
> 我们注意到，也有类似的报道，在[29]中报告说非线性是有帮助的，然而却将传统残差块输入中的非线性移除，从而提高了CIFAR数据集的性能。(言外之意就是说：非线性确实有帮助，但在[29]中还是将其移除了，而移除了效果会更好)

--------
<br>

This is in contrast with traditional convolutional blocks, both regular and separable, where both expressiveness and capacity are tangled together and are functions of the output layer depth.

&emsp; &emsp; From `MobileNetV2: Inverted Residuals and Linear Bottlenecks`

> TIPS: \
> expressiveness  n. 表达能力;善于表现;表情丰富
> 
> Translate: \
> 这与传统的卷积块形成对比，无论是规则块还是可分块，在传统卷积块中，表达性和容量都是纠缠在一起的，是输出层深度的函数。


--------
<br>









