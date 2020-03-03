---
title: 
date: 2018-09-14
tags:
categories: ["深度学习笔记"]
mathjax: true
---

论文：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, NIPS 2015

<!-- more -->

SSD 是 2016年发表，SSD没有全连接层，FasterRCNN有全连接层。

SSD和FasterRCNN在Feature map区域建议采样方式的区别：SSD是在6个feature map上用了4~6个anchor框去做采样（sliding widow依然是3x3），而FasterRCNN是在最后一个feature map上用9个anchor框去采样。所以SSD的采样框比FasterRCNN多，进而精度也比FasterRCNN高，但其抛弃了ROI pooling，所以其稳定性没有FasterRCNN高。
这里的精度和稳定性是这样理解的：检测某种物体很准确，但对检测另一种物体不准确；这段时间检测准确性很好，过一段时间后检测准确性又不行了。


# Abstract
最先进的目标检测网络依赖于区域建议算法来假设目标的位置。SPPnet[1]和快速R-CNN[2]等技术的进步缩短了检测网络的运行时间，暴露了区域建议计算的瓶颈。在这项工作中，我们引入了一个区域建议网络(RPN)，它与检测网络共享全图像卷积特性，从而实现了几乎免费的区域建议。RPN是一个全卷积网络，它同时预测每个位置的对象界限和对象得分。RPN是经过en训练的


