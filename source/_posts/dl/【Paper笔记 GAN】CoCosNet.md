---
title: 
date: 2020-7-15
tags:
categories: ["深度学习笔记"]
mathjax: true
---

Cross-domain Correspondence Learning for Exemplar-based Image Translation \
Pan Zhang1 ∗, Bo Zhang2, Dong Chen2, Lu Yuan3, Fang Wen2 \
1 University of Science and Technology of China \
2 Microsoft Research Asia \
3 Microsoft Cloud+AI

CVPR2020

<!--more-->


**Abstract**
&emsp; We present a general framework for exemplar-based image translation, which synthesizes a photo-realistic image from the input in a distinct domain (e.g., semantic segmentation mask, or edge map, or pose keypoints), **given an exemplar image**. The output has the style (e.g., color, texture) in consistency with the **semantically corresponding objects in the exemplar**. 

We propose to jointly learn the cross-domain correspondence and the image translation, where both tasks facilitate each other and thus can be learned with weak supervision. \
我们提出“cross-domain correspondence”和“image translation”的联合学习，这两个任务相互促进，因此可以在弱监督的情况下学习。

The images from distinct domains are **first aligned to an intermediate domain** where dense correspondence is established. \
首先将来自不同domain的图片对齐到中间domain（这个中间domain是密集对齐的）。

Then, the network **synthesizes images based on the appearance of semantically corresponding patches in the exemplar**. \
然后，network 根据 exemplar 中对应语义 patches 的外观来合成图片。

We demonstrate the effectiveness of our approach in several image translation tasks. Our method is superior to state-of-the-art methods in terms of image quality significantly, with the image style faithful(adj. 忠实的，忠诚的；如实的；准确可靠的) to the exemplar with semantic consistency. Moreover, we show the utility of our method for several applications.


# Introduction


# Related Work


# Approach
&emsp; We aim to learn the translation from the source domain A to the target domain B given an input image xA ∈ A and an exemplar image yB ∈ B. The generated output is desired to conform(vt.使一致,遵守,使顺从; vi.一致,符合) to the content as xA while resembling the style from semantically similar parts in yB. For this purpose, **the correspondence** between $x_A$ and $y_B$ , which lie in different domains, **is first established**, and **the exemplar image is warped** accordingly so that its semantics is aligned with $x_A$ (Section 3.1). **Thereafter, an image is synthesized according to the warped exemplar** (Section 3.2). The whole network architecture is illustrated in Figure 2, by the example of mask to image synthesis.


## Cross-domain correspondence network
&emsp; Usually the semantic correspondence is found by matching patches [27, 25] in the feature domain with a pre-trained classification model. However, pre-trained models are typically trained on a specific type of images, e.g., natural images, so the extracted features cannot generalize to depict the semantics for another domain. Hence, prior works cannot establish the correspondence between heterogeneous(adj.异种的, 异质的, 由不同成份形成的) images, e.g., edge and photo-realistic images. To tackle this, **we propose a novel cross-domain correspondence network, mapping the input domains to a shared domain** $S$ in which the representation is capable to represent the semantics for **both** input domains. As a result, reliable semantic correspondence can be found within domain $S$.

> 通常的 semantic correspondence 是通过预训练模型提取的特征来做的，但是预训练模型所使用的domain和我们的目标domain往往差异很大。为了解决这个问题，我们提出了一个新颖的 “cross-domain correspondence network”


**Domain alignment** As shown in Figure 2, we first adapt the input image and the exemplar to a shared domain $S$. `To be specific (具体地说)`, xA and yB are fed into the feature pyramid network that extracts multi-scale deep features by leveraging(laverage n.杠杆作用, 影响力; v.举债经营, 贷款投机) both local and global image context [41, 28]. The extracted feature maps are further transformed to the representations in $S$, denoted by $x_S \in \mathbb{R}^{HW \times S}$ and $y_S \in \mathbb{R}^{HW \times S}$ respectively ($H,W$ are feature spatial size; $C$ is the channel-wise dimension). Let $\mathcal{F}_{A→S}$ and $\mathcal{F}_{B→S}$ be the domain transformation from the two input domains respectively, so the adapted representation can be formulated as,

公式(1,2) 见 paper

where θ denotes the learnable parameter. The representation $x_S$ and $y_S$ comprise(vt.包含；由…组成) discriminative features that characterize the semantics of inputs. Domain alignment is, in practice, essential for correspondence in that only when $x_S$ and $y_S$ reside in the same domain can they be further matched with some similarity measure.



**Correspondence within shared domain** We propose to match the features of xS and yS with the correspondence layer proposed in [49]. Concretely, **we compute a correlation matrix** $\mathcal{M} ∈ \mathbb{R} ^{HW×HW}$ of which each element is a pairwise feature correlation,

公式(3) 见 paper

where $\hat{x}_S(u)$ and $\hat{y}_S(v) \in \mathbb{R}^C$ represent the channel-wise centralized feature of $x_S$ and $y_S$ in position $u$ and $v$, i.e., $\hat{x}_S (u) = x_S (u) - \text{mean} (x_S(u))$ and $\hat{y}_S (v) = y_S (v) - \text{mean} (y_S (v))$. $\mathcal{M}(u, v)$ indicates a higher semantic similar- ity between $x_S (u)$ and $y_S (v)$.


&emsp; Now the challenge is **how to learn the correspondence without direct supervision**. Our idea is to **jointly train with image translation**. The translation network may find it easier to generate high-quality outputs only by referring to the correct corresponding regions in the exemplar, which implicitly(含蓄地;暗中地;隐式地) pushes the network to learn the accurate correspondence. In light of this, we warp yB according to $\mathcal{M}$ and obtain the warped exemplar $r_{y → x} \in \mathbb{R}^{HW}$. **Specifically, we obtain $r_{y→x}$ by selecting the most correlated pixels in $y_B$ and calculating their weighted average,** \
具体来说，我们通过在 $y_B# 中选择最相关的像素并计算其加权平均值来获得 $r_{y→x}$. \

公式(4) in paper

Here, α is the coefficient that controls the sharpness of the softmax and we set its default value as 100. In the following, images will be synthesized conditioned on $r_{y→x}$ and the correspondence network, in this way, learns its assignment with indirect supervision.


未完待续。。。


## Translation network







