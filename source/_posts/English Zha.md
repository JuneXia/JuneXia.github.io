---
title: 
date: 2019-06-25
tags:
categories: ["深度学习笔记"]
mathjax: true
---

In a convolutional fashion, we evaluate a small set (e.g. 4) **of** default boxes **of** different aspect ratios at each location in several feature maps with different scales. 
&emsp; &emsp; `From SSD:Single Shot MultiBox Detector`

> TIPS: 
> v. &ensp;n1 of n2 of n3  → v. &ensp;n1 n3 的 n2 \
> `in several feature maps with different scales. ` 在几个不同尺度的feature map中
> 
> Translate:
使用卷积的方法，我们在几个不同尺度的feature map中的每个位置  评估一小组(e.g. 4)不同长宽比的default boxes.

--------
<br>

To achieve high detection accuracy we **produce** predictions **of** different scales **from** feature maps **of** different scales, and explicitly separate predictions by aspect ratio. 
&emsp; &emsp; `From SSD:Single Shot MultiBox Detector`

> TIPS:
> v. &ensp;n1 of n2 from n3 of n4  →  n4 的 n3 &ensp;v. &ensp;n2 的 n1
> 
> Translate:
> 为了实现较高的检测精度，我们从不同尺度的特征图中生成不同尺度的预测，并通过纵横比明确地分离预测。

--------
<br>

Down sampling **is handled with** strided convolution **in** the depthwise convolutions **as well as in** the first layer.
&emsp; &emsp; `MobileNets: Efficient Convolutional Neural Networks for Mobile Vision`

> TIPS: \
> n1 **be v.ed with** n2 **in** n3 **as well as in** n4. → n1 **在** n3 **和** n4 **中都 v.ed with** n2.
> 
> Translate:
> 下行采样在深度卷积和第一层卷积中都使用了strided convolution.

--------
<br>


When training MobileNets we **do not use** side heads or label smoothing **and** additionally reduce the amount image **of** distortions **by** limiting the size **of** small crops **that** are used in large Inception training. 
&emsp; &emsp; `MobileNets: Efficient Convolutional Neural Networks for Mobile Vision`

> TIPS:
> additionally &ensp; adv. 此外；又，加之
> 
> Translate
> 当训练MobileNet的时候，我们不使用side heads或者标签平滑，并且通过限制small crops的尺寸来减少失真图片数量，这个small crops在大型 Inception 训练中会被用到。

--------
<br>





