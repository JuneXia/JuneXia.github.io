---
title: 
date: 2020-6-20
tags:
categories: ["深度学习笔记"]
mathjax: true
---
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) \
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros \
Berkeley AI Research (BAIR) Laboratory, UC Berkeley \
{isola,junyanz,tinghuiz,efros}@eecs.berkeley.edu

CVPR2017

**Abstract**
&emsp; We investigate conditional adversarial networks as a `general-purpose(通用的)` solution to image-to-image translation problems. These networks **not only** learn the mapping from input image to output image, **but also learn a loss function to train this mapping**. This makes it possible to apply the same generic(adj.一般的,通用的;属的;非商标的) approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking(tweak v.扭,捏,拧;稍稍改进,对…稍作调整). `As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable(adj.合理的,公道的;通情达理的) results without hand-engineering our loss functions either. (作为一个社区，我们不再手工设计我们的映射函数，而这项工作表明，我们也可以无需手工设计我们的损失函数就可以实现理想的结果).`
> 我们使用Conditional-GAN来作为image-to-image的解决方案，我们的网络不仅会学习input到output的映射，也会学习训练这个映射的损失函数。

# Introduction
&emsp; Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input image into a corresponding output image. `Just as a concept may be expressed in either English or French (正如一个概念既可以用英语也可以用法语表达一样)`, a scene may be rendered as an RGB image, a gradient field, an edge map, a semantic label map, etc. In analogy to automatic language translation, we define automatic image-to-image translation as the task of translating one possible representation of a scene into another, given sufficient training data (see Figure 1). Traditionally, each of these tasks has been tackled(tackle 解决,处理,对付) with separate, special-purpose machinery (e.g., [16, 25, 20, 9, 11, 53, 33, 39, 18, 58, 62]), despite the fact that the setting is always the same: predict pixels from pixels. Our goal in this paper is to develop a common framework for all these problems.
> summary: 就像language translation(语言翻译)一样，我们这里定义image-to-image 的转换问题。基本都是废话。

》 未完待续。。。

&emsp; The community has already taken significant steps in this direction, with convolutional neural nets (CNNs) becoming the common workhorse behind a wide variety of image prediction problems. CNNs learn to minimize a loss function – an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still goes into designing effective losses. In other words, we still have to tell the CNN what we wish it to minimize. But, just like King Midas, we must be careful what we wish for! If we take a naive approach and ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results [43, 62]. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring. Coming up with loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images – is an open problem and generally requires expert knowledge.















