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

&emsp; In the ImageNet-1K dataset, we obtain accuracy improvement from various baseline networks by plugging(plug n.插头;塞子;栓;v.插入;塞住) our tiny module, revealing the efficacy of CBAM. We visualize trained models using the grad-CAM [18] and observe that CBAM-enhanced networks focus on target objects more properly than their baseline networks. Taking this into account, we conjecture(n/v.推测;猜想) that the performance boost comes from accurate attention and noise reduction of irrelevant clutters(clutter n/v.杂乱,混乱). Finally, we validate performance improvement of object detection on the MS COCO and the VOC 2007 datasets, demonstrating a wide applicability of CBAM. Since we have carefully designed our module to be light-weight, the overhead(n.日常开支,运营费用;adj.在头上方的,在空中的) of parameters and computation is negligible(adj.微不足道的,可以忽略的) in most cases.

**Contribution.** Our main contribution is three-fold. 
1. We propose a simple yet effective attention module (CBAM) that can be widely applied to boost representation power of CNNs.
2. We validate the effectiveness of our attention module through extensive ablation(n.[水文]消融;切除) studies. 
3. We verify that performance of various networks is greatly improved on the multiple benchmarks (ImageNet-1K, MS COCO, and VOC 2007) by plugging our light-weight module.


# Related Work
待续。。。

# Convolutional Block Attention Module
Given an intermediate feature map $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ as input, CBAM sequentially infers a 1D channel attention map $\mathbf{M_c} \in \mathbb{R}^{C \times 1 \times 1}$ and a 2D spatial attention map $\mathbf{M_s} \in \mathbb{R}^{1 \times H \times W}$ as illustrated in Fig. 1. The overall attention process can be summarized as:
$$
\begin{aligned}
    \mathbf{F' = M_c(F) \otimes F}, \\
    \mathbf{F'' = M_c(F') \otimes F'}, 
\end{aligned} \tag{1}
$$
where $\otimes$ denotes **element-wise multiplication**. During multiplication, the attention values are broadcasted (copied) accordingly: channel attention values are broadcasted along the spatial dimension, and `vice versa(反之亦然)`. $F''$ is the final refined output. Fig. 2 depicts the computation process of each attention map. The following describes the details of each attention module.

**Channel attention module.** We produce a channel attention map by exploiting the inter-channel relationship of features. As each channel of a feature map is considered as a feature detector [31], channel attention focuses on what is meaningful given an input image. **To compute the channel attention efficiently, we squeeze the spatial dimension of the input feature map.** For aggregating(v.聚集;合计) spatial information, average-pooling has been commonly(adv.一般地;通常地;普通地) adopted `so far(迄今为止)`. Zhou et al. [32] suggest to use it to learn `the extent of(在…的范围内;到…的程度)` the target object effectively and Hu et al. [28] adopt it in their attention module to compute spatial statistics. Beyond the previous works, we argue that **max-pooling gathers another important clue about distinctive(adj.独特的,有特色的) object features to infer finer(adj.更好的;更优质的) channel-wise attention**. Thus, **we use both average-pooled and max-pooled features simultaneously.** We empirically(adv.以经验为主地) confirmed that exploiting both features greatly improves representation power of networks rather than using each independently (see Sec. 4.1), showing the effectiveness of our design choice. We describe the detailed operation below.

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/CMAB1.jpg" width = 70% height = 70% />
</div>

&emsp; We first aggregate spatial information of a feature map by using both averagepooling and max-pooling operations, generating two different spatial context descriptors: $\mathbf{F^c_{avg}}$ and $\mathbf{F^c_{max}}$, which denote average-pooled features and max-pooled features respectively. Both descriptors are then forwarded to a shared network to produce our channel attention map $\mathbf{M_c} \in \mathbb{R}^{C \times 1 \times 1}$. \
对$\mathbf{M_c} \in \mathbb{R}^{C \times 1 \times 1}$的理解：
如图Fig.2右侧的 $\mathbb{M_c}$，其参数属于实数空间$\mathbb{R}$，$C$ 是其channle数量，$1 \times 1$ 是空间尺寸。

The shared network is composed of multi-layer perceptron (MLP) with one hidden layer. To reduce parameter overhead, the hidden activation size is set to $\mathbb{R}^{C / r \times 1 \times 1}$, where $r$ is the reduction ratio. \
对$\mathbb{R}^{C / r \times 1 \times 1}$的理解： \
应看成 $\mathbb{R}^{(C / r) \times 1 \times 1}$，即$C/r$ 是一个整体，表示对 C 衰减 r 倍，如图Fig.2中间的 Shared MLP，对于 MLP 的 hidden-layer，其输入channel数量是 $C$，输出通道数量是 $C/r$，$1 \times 1$ 是空间尺寸（将其看成是一个 $1 \times 1$ 的 feature-map）.

After the shared network is applied to each descriptor, we merge the output feature vectors using element-wise summation. In short, the channel attention is computed as:
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/CMAB2.jpg" width = 70% height = 70% />
</div>

where σ denotes the sigmoid function, $\mathbf{W_0} \in \mathbb{R}^{C/r \times C}$, and $\mathbb{R}^{C \times C / r}$. Note that the MLP weights, $\mathbf{W_0}$ and $\mathbf{W_1}$, are shared for both inputs and the ReLU activation function is followed by $\mathbf{W_0}$.

对公式(2)的理解： \
假设 $\mathbf{F}$ 表示是一个已经pooling过尺寸为$C \times 1$的特征，$\mathbf{W_0 \times F}$ 就得到一个尺寸为 $(C/r) \times 1$ 的特征 $\mathbf{F_1}$，然后 $\mathbf{W_1 \times F_1}$ 就得到一个尺寸为 $C \times 1$ 的特征 $\mathbf{F_2}$.

主要思想已经讲完，其他待续。。。

