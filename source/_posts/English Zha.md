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

<br>

To achieve high detection accuracy we produce predictions of different scales from feature maps of different scales, and explicitly separate predictions by aspect ratio. 
&emsp; &emsp; `From SSD:Single Shot MultiBox Detector`



> TIPS:
> v. &ensp;n1 of n2 from n3 of n4  →  n4 的 n3 &ensp;v. &ensp;n2 的 n1
> 
> Translate:
> 为了实现较高的检测精度，我们从不同尺度的特征图中生成不同尺度的预测，并通过纵横比明确地分离预测。


