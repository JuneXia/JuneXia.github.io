---
title: 
date: 2020-04-30
tags:
categories: ["深度学习笔记"]
mathjax: true
---
Squeeze-and-Excitation Networks
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
16 May 2019
<!-- more -->

**Abstract** The central building block of convolutional neural networks (CNNs) is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. A broad range of prior research has investigated the spatial component of this relationship, seeking to strengthen the representational power of a CNN by enhancing the quality of spatial encodings throughout its feature hierarchy(n.层级;等级制度). In this work, **we focus instead on the channel relationship and propose a novel architectural unit, which we term the Squeeze-and-Excitation(n.激发,刺激;激励;激动) (SE) block**, `that adaptively recalibrates(重新校正) channel-wise feature responses by explicitly(adv.明确地;明白地) modelling interdependencies(n.互相依赖;相关性) between channels. (它通过明确地建模通道之间的相互依赖关系，自适应地重新校准通道方向的特征响应).` \
We show that these blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. We further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight(n/v.怠慢;轻蔑;adj.轻微的,少量的;脆弱的;细长的) additional computational cost. Squeeze-and-Excitation Networks formed the foundation of our ILSVRC 2017 classification submission which won first place and reduced the top-5 error to 2:251%, surpassing(surpass v.优于,超出,胜过) the winning entry of 2016 by a relative improvement of 25%. Models and code are available at https://github.com/hujie-frank/SENet.

**Index Terms**—Squeeze-and-Excitation, Image representations, Attention, Convolutional Neural Networks.


# Introduction
从这里开始...未完待续。。。
Convolutional neural networks(CNNs) have proven to be useful models for tackling a wide range of visual tasks [1], [2], [3], [4]. At each convolutional layer in the network, a collection of filters expresses neighbourhood spatial connectivity patterns along input channels fusing spatial and channel-wise information together within local receptive fields. By interleaving a series of convolutional layers with non-linear activation functions and downsampling operators, CNNs are able to produce image representations that capture hierarchical patterns and attain global theoretical receptive fields. A central theme of computer vision research is the search for more powerful representations that capture only those properties of an image that are most salient for a given task, enabling improved performance. As a widely-used family of models for vision tasks, the development of new neural network architecture designs now represents a key frontier in this search. Recent research has shown that the representations produced by CNNs can be strengthened by integrating learning mechanisms into the network that help capture spatial correlations between features. One such approach, popularised by the Inception family of architectures [5], [6], incorporates multi-scale processes into network modules to achieve improved performance.













