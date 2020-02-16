---
title: 【深度学习笔记 paper】CosFace
date: 2019-10-13 17:28:05
tags:
categories: ["深度学习笔记"]
mathjax: true
---

论文：[CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)
<!-- more -->

&emsp; 人脸识别Loss函数的优化思想同之前介绍过的 Center Loss 类似，即希望训练得到的人脸特征是具有辨识力的，也就是  maximizing inter-class
variance and minimizing intra-class variance(类内紧类间开)。H.Wang 等人2018年在论文 CosFace 中提出的 Cosine Loss 是从角度空间去思考优化Softmax，计算公式如下：

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/cosface1.jpg" width = 80% height = 80% />
</div>

作者认为softmax中的 
$\boldsymbol{Wx}$ 
实际可以从几何角度 
$\boldsymbol{Wx} =  \Vert \boldsymbol{W} \Vert \Vert \boldsymbol{x} \Vert cos(\boldsymbol{\theta})$
去理解，所以对 
$\boldsymbol{W}$ 和 $\boldsymbol{x}$ 
做完归一化处理后，
$\boldsymbol{Wx}$ 实际上可以看成是对应类别的余弦相似度，那么 Cosine Loss 则希望预测正确的类别相似度要比其他类别要大的多，即使是减去一个 margin 后还要比其他类别大，这就是本篇论文的核心思想。
