---
title: 
date: 2019-07-05
tags:
categories: ["深度学习笔记"]
mathjax: true
---



**Image pyramid v.s. feature pyramid:** 在 sliding window paradigm 中，一个分类器被应用在一个dense image grid上，这可以追溯到过去的几十年。里程碑式的工作是 Viola-Jones [Robust real-time face detection. IJCV2004] 的探索了用cascade chain实时有效地从图像金字塔中剔除 false face 区域，这导致了尺度不变的人脸检测框架被广泛采用 [如MTCNN、Joint cascade face detection and alignment ECCV2014]. 尽管图像金字塔上的滑动窗口是主要的检测范式[19,32]，但随着特征金字塔[Feature pyramid networks for object detection.CVPR2017]的出现，多尺度特征图[S3fd,Pyramidbox]上的 sliding-anchor [Faster r-cnn] 迅速主导了人脸检测。

**Two-stage v.s. single-stage:** 目前的人脸检测方法继承了通用目标检测方法的一些成果，可分为两类：Two-stage 方法(如Faster R-CNN[43, 63, 72])和 single-stage 方法(如SSD[30, 68]和RetinaNet[29, 49])。Two-stage 方法采用“建议和细化”机制，具有较高的定位准确性。相比之下，single-stage 方法密集采样人脸位置和尺度，导致训练过程中正负样本极不平衡。为了解决这种不平衡，sampling [47]和 re-weighting [29]的方法被广泛采用。与两阶段方法相比，单阶段方法效率更高且召回率更高，但存在FPR更高和降低定位accuracy的风险。

**Context Modelling 上下文建模:** 为了增强模型的上下文推理能力（这有助于捕获小面孔）[23]，SSH[36]和PyramidBox[49]在特征金字塔上应用上下文模块，扩大了Euclidean grids的receptive field。为了提高CNN的non-rigid transformation(非刚性转换) 建模能力，可变形卷积网络(deformable convolution network, DCN)[9,74]采用了一种新的可变形layer来建模geometric transformations(几何变换)。“WIDER Face Challenge 2018”[33] 的冠军解决方案指出，刚性(expansion, 扩展)和非刚性(deformation, 变形)上下文建模是互补和正交的，以提高人脸检测的性能。

**Multi-task Learning:** “Joint face detection and alignment” 被广泛使用[6,66,5]，因为对齐的人脸为face classification提供了更好的特征。在Mask R-CNN[20]中，通过添加一个分支来预测对象掩码（与现有分支并行），大大提高了检测性能。Densepose[1]采用了Mask-RCNN的架构，在每个选定的区域内获得dense部分的标签和坐标。然而，文献[20,1]中的dense regression分支是通过监督学习进行训练的。此外，dense分支是一个小的FCN应用于每个RoI，以预测pixel-to-pixel的dense映射。

# RetinaFace

## Multi-task Loss