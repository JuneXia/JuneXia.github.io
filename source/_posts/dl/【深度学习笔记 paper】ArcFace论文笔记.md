---
title: 【深度学习笔记 paper】ArcFace论文笔记
date: 2019-10-21 17:28:05
tags:
categories: ["深度学习笔记"]
mathjax: true
---

 ## ArcFace: Additive Angular Margin Loss for Deep Face Recognition


# Abstract
&emsp; 在使用DCNN做大规模人脸识别中最主要的挑战是设计恰当的损失函数以提高识别辨识力。最近，一个流行的研究方向是将margin并入损失函数以最大化人脸类别可分性。为了在人脸识别中获得具有高辨识力的特征，本文我们提出了Additive Angular Margin Loss (ArcFace)。ArcFace因为与超球面上的geodesic distance具有精确的对应关系，所以它具有清晰的几何解释。
<!-- more -->

# 1. Introduction
&emsp; 使用DCNN做人脸识别有两个主要的research line，其中一个就像是使用softmax训练一个多类别分类器一样，它能有效分类训练集中的不同identities；另一个是直接学习一个embedding，例如 triplet loss。基于大规模的训练数据集和精心设计的DCNN结构，softmax方法和triplet方法都能在人脸识别上获得卓越的表现。然而softmax和triplet都有一些缺点。
对于softmax loss来说：
（1）linear transformation matrix的尺寸随identity的数量呈线性增加；
（2）学习到的特征对closed-set闭集分类问题来说是separable的，但是对于open-set开集的人脸识别问题来说却没有足够的discriminative。
对于triplet loss来说：
（1）face triplets 数量是一个组合爆炸，尤其是对大规模数据集来说，这将导致迭代次数显著增加；
（2）semi-hard样本挖掘是一个相当困难的问题。


&emsp; 一些变体 [38, 9, 46, 18, 37, 35, 7, 34, 27] 被提出以改进softmax loss的辨识力度。Wen 等人首先提出了center loss，即feature vector和它的类别中心之间的欧式距离，为了获得类内紧致类间分散的保证，他们使用center loss和softmax loss的联合惩罚。

&emsp; 注意到通过softmax loss训练出来的DCNN的最后一个全连接层的权重与face类别的中心具有概念上的相似性，SphereFace和L-Softmax提出一个multiplicative angular margin penalty来同时enforce intra-class（类内）紧致性和inter-class（类间）差异性。尽管SphereFace引入了angular margin这一重要思想，然而为了能够被计算，他们的loss函数被要求做一系列的approximations（近似），这将导致网路训练不稳定。为了能够稳定训练，他们又提出了一个混合loss函数，这其中包括标准的softmax loss。经验表明，softmax loss在训练过程中占主导地位，因为基于积分的multiplicative angular margin 使得targit logit曲线非常陡峭，从而阻碍了收敛<font color=red>（不知所云）</font>。CosFace直接为targit logit增加cosine margin惩罚，这相比于SphereFace能够获得更好的performance，但CosFace允许更容易的实验并且摆脱了使用需要softmax loss的联合监督。

&emsp; 在本文中，我们提出了Additive Angular Margin Loss (ArcFace) 来更进一步地改善人脸识别模型的辨识力度并且稳定其训练过程。如图2所示，DCNN的feature和最后一个全连接层分别normalisation后再进行点乘，这等于cosine距离。我们使用arccos函数计算当前feature和target weight之间的角度。然后，我们增加一个附加angular margin到target angle，我们通过cosine函数获得目标logit。然后，我们通过固定的feature norm来re-scale所有的logits，随后的步骤就和softmax loss极其相似了。ArcFace的优点总结如下：
Engaging .....
Effective ....
Easy ....
Efficient ....

# Proposed Approach
## 2.1 ArcFace

## 2.2 Comparison with SphereFace and CosFace
**Numerical Similarity.** 在 SphereFace, ArcFace和CosFace中，分别有三个不同的margin penalty被提出，即 multiplicative angular margin $m_1$, additive angular margin $m_2$, and additive cosine margin $m_3$. 从数值分析来看，对于不同的margin penalties，无论它是被添加到angle space 还是 cosine space，通过惩罚target logit它们都加强了 intra-class 紧致和 inter-class分散。


**Geometric Difference.** 尽管ArcFace和之前的一些works具有numerical similarity，但是我们提出的 additive angular margin 具有更好的几何性质，因为angular margin与geodesic distance有更准确一致性。

## 2.3 Comparison with Other Losses
**Intra-Loss**

**Inter-Loss**

**Triplet-Loss**


# 3. Experiments
## 3.1 Implementation Details

## 3.2 Ablation Study on Losses
……

除此之外，我们还与其他基于margin的方法进行了比较，我们进一步比较了ArcFace与其他损失



