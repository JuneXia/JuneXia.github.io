---
title: 
date: 2020-04-27
tags:
categories: ["深度学习笔记"]
mathjax: true
---

CBAM: Convolutional Block Attention Module

Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon

Korea Advanced Institute of Science and Technology, Daejeon, Korea
{shwoo93, iskweon77}@kaist.ac.kr

Lunit Inc., Seoul, Korea
jcpark@lunit.io

Adobe Research, San Jose, CA, USA
jolee@adobe.com

<!-- more -->

**Abstract**
We propose Convolutional Block Attention Module (CBAM), a simple yet effective attention module for feed-forward convolutional neural networks. **Given an intermediate feature map, our module sequentially(adv.从而;继续地;循序地) infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to the input feature map for adaptive feature refinement(n.改进,改善;精炼;细化)**. 
如Fig.1所示

Because CBAM is a **lightweight and general module**, it can be integrated into any CNN architectures seamlessly(adv.无缝地) with negligible(adj.微不足道的,可以忽略的) overheads(日常开支;一般费用) and is end-to-end trainable along with base CNNs. We validate our CBAM through extensive experiments on ImageNet-1K, MS COCO detection, and VOC 2007 detection datasets. Our experiments show consistent improvements in classification and detection performances with various models, demonstrating the wide applicability of CBAM. The code and models will be publicly available.

**Keywords**: Object recognition, attention mechanism, gated convolution

# Introduction
Convolutional neural networks (CNNs) have significantly pushed the performance of vision tasks [1-3] based on their rich representation power. To enhance performance of CNNs, **recent researches** have mainly investigated three important factors of networks: **depth, width, and cardinality**([数]基数,(集的)势).

&emsp; From the LeNet architecture [4] to Residual-style Networks [5-8] so far, the network has become deeper for rich representation. VGGNet [9] shows that stacking blocks with the same shape gives fair results. Following the same spirit, **ResNet** [5] stacks the same topology of residual blocks along with skip connection to build an extremely **deep** architecture. **GoogLeNet** [10] shows that **width** is another important factor to improve the performance of a model. Zagoruyko and Komodakis [6] propose to increase the width of a network based on the ResNet architecture. They have shown that a 28-layer ResNet with increased width can outperform an extremely deep ResNet with 1001 layers on the CIFAR benchmarks. **Xception** [11] and **ResNeXt** [7] come up with to increase the **cardinality** of a network. They empirically(adv.以经验为主地;经验主义地) show that cardinality not only saves the total number of parameters but also results in stronger representation power than the other two factors: depth and width.

&emsp; `Apart from(远离,除…之外;且不说)` these factors, we investigate a different aspect of the architecture design, attention. The significance of attention has been studied extensively in the previous literature [12-17]. Attention not only tells where to focus, it also improves the representation of interests. **Our goal is to increase representation power by using attention mechanism: focusing on important features and suppressing unnecessary ones**. In this paper, we propose a new network module, named “Convolutional Block Attention Module”. Since convolution operations extract informative features **by blending(n.混合;调配;混和物;v.混合;协调) cross-channel and spatial information together, we adopt our module to emphasize meaningful features along those two principal dimensions: channel and spatial axes**. To achieve this, we sequentially apply channel and spatial attention modules (as shown in Fig. 1), so that each of the branches can learn what and where to attend in the channel and spatial axes respectively. As a result, our module efficiently helps the information flow within the network by learning which information to emphasize or suppress.

&emsp; In the ImageNet-1K dataset, we obtain accuracy improvement from various baseline networks by plugging our tiny module, revealing the efficacy of CBAM. We visualize trained models using the grad-CAM [18] and observe that CBAMenhanced networks focus on target objects more properly than their baseline networks. Taking this into account, we conjecture that the performance boost comes from accurate attention and noise reduction of irrelevant clutters. Finally, we validate performance improvement of object detection on the MS COCO and the VOC 2007 datasets, demonstrating a wide applicability of CBAM. Since we have carefully designed our module to be light-weight, the overhead of parameters and computation is negligible in most cases.




