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


This structure maintains a compact representation at the input and the output **while** expanding to a higher-dimensional feature space internally to increase the expressiveness of nonlinear perchannel transformations.

&emsp; &emsp; From `Searching for MobileNetV3`

> Translate: \
> 这种结构在输入和输出处保持了一种紧凑的表示，**同时**在内部扩展到高维特征空间，以增加每个通道的非线性转换的表达能力。

--------
<br>

MnasNet built upon the MobileNetV2 structure **by introducing** lightweight attention modules **based on** squeeze and excitation **into** the bottleneck structure. \
&emsp; &emsp; From `Searching for MobileNetV3`

> TIPS: \
> excitation  n. 激发，刺激；激励；激动
> 
> Translate: \
> MnasNet建立在MobileNetV2结构上，通过在瓶颈结构中引入基于挤压和激励的轻量级注意模块。

--------
<br>

It outperforms other detection methods, including DPM and R-CNN, when **generalizing from** natural images **to** other domains like artwork. \
&emsp; &emsp; From `You Only Look Once:Unified, Real-Time Object Detection`

> TIPS: \
> generalizing : 归纳阶段,归纳,形成概念
> generalizing from ...: 从...归纳出
> generalizing from ... to ...: 从...推广到...
> 
> Translate: \
> 当从自然图像推广到艺术作品等其他领域时，它的性能优于其他检测方法，包括 DPM 和 R-CNN。

--------
<br>

Fast, accurate algorithms for object detection would allow computers to drive cars without specialized sensors, **enable** `assistive` devices **to** `convey` real-time scene information to human users, and unlock the potential for `general purpose`, `responsive` robotic systems. \
&emsp; &emsp; From `You Only Look Once:Unified, Real-Time Object Detection`

> TIPS: \
> general n. 一般；将军，上将；常规; 
        adj. 一般的，普通的；综合的；大体的
> purpose n. 目的；用途；意志;
         vt. 决心；企图；打算
> general purpose: 通用的
> responsive: adj.响应的;应答的;响应灵敏的
> assistive: 辅助的
> devices to convey: vt.传达;运输
> 
> 快速、准确的目标检测算法将允许计算机在没有专门传感器的情况下驾驶汽车，**使**辅助设备**能够向**人类用户传递实时的场景信息，并为通用、响应灵敏的机器人系统释放潜力。

--------
<br>

Our network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes **across** all classes for an image simultaneously(adv.同时地). \
&emsp; &emsp; From `You Only Look Once:Unified, Real-Time Object Detection`
> TIPS: \
> across &emsp; adv. 从……的一边到另一边，穿过，越过；在对面，横过；宽；向；（纵横填字游戏）横向字谜答案 \
> prep. 从……的一边到另一边，穿过；在……对面，另一边；在……上；在各处，遍及；在……里
> 
> Translate: \
> 我们的网络使用整个图像的特征来预测每个边界框。它还可以同时预测一个图像在所有类**中的**所有边界框。

--------
<br>



--------
<br>



--------
<br>



--------
<br>


--------
<br>
