---
title: 
date: 2018-06-20
tags:
categories: ["深度学习笔记"]
mathjax: true
---

论文：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
*(Submitted on 18 Jun 2014 ([v1](https://arxiv.org/abs/1406.4729v1)), last revised 23 Apr 2015 (this version, v4))*，2015年发表于IEEE.
<!-- more -->

TODO：已经有几张截图位于路径下，但还未贴出。

**问题背景**：因为RCNN要求FC的长度是固定的，所以要求卷积层的输出也是固定的，近而就要求Input Image 尺寸是固定的，这就需要对proposal 区域进行crop或者warp操作（warp实际就是resize）。但crop会造成信息丢失，warp会造成图片变形。

简单来说，SPP-net 就是为了解决卷积网络能够接受任意尺寸的图片保留原始信息而提出的。


**Abstract**
&emsp; 现有的深度卷积神经网络(CNNs)需要一个固定大小的输入图像(e.g., 224 224)。这种 requirement 是人为的，可能会降低对任意size/scale(大小/尺度、比例)的images或sub-images的识别accuracy。在这项工作中，我们为网络配备了另一种pooling策略：“spatial pyramid pooling”（空间金字塔池化），以消除上述 requirement. 这种新的网络结构被称为SPP-net，它可以生成固定长度的 representation，而不管图像的大小和比例。金字塔池对物体 deformations（变形）也有很强的鲁棒性。基于这些优点，一般来说，SPP-net 应该可以改进所有基于cnn的图像分类方法。在ImageNet 2012数据集上，我们证明了SPP-net 提高了各种CNN架构的准确性，尽管它们的设计不同。在Pascal VOC 2007和Caltech101数据集上，SPP-net 在使用单一的 full-image representation 且没有 fine-tuning 的情况下实现了 state-of-the-art 分类结果。

&emsp; SPP-net 的能力在目标检测中也很重要。使用SPP-net，我们从entire image（整个图像）计算feature map 并且只计算一次，然后在任意区域(sub-images)中pool(合并;聚集;池化) feature来生成固定长度的representation来训练检测器。该方法避免了卷积特征的重复计算。在处理测试图像时，我们的方法比R-CNN快24~102倍，同时在Pascal VOC 2007上获得更好的或comparable（可比较的，相当的）的accuracy.

&emsp; 在2014年的ImageNet Large Scale Visual Recognition Challenge (ILSVRC)中，我们的方法在所有38个团队中取得了obeject detection排名第二、image classification排名第三的成绩。本文还介绍了本次比赛的改进情况。


# Introduction
&emsp; 我们的视觉社区正在经历一场快速的、革命性的变化，这主要是由深度卷积神经网络(CNNs)[1]和大规模训练数据[2]的可用性引起的。最近，基于深度网络的方法在image classification[3]、[4]、[5]、[6]、object detection[7]、[8]、[5]以及许多其他识别任务[9]、[10]、[11]、[12]上都有了实质上的改进，甚至在非识别任务技术上也有了很大的改进。

&emsp; However, there is a technical issue in the training and testing of the CNNs: the prevalent(普遍的,流行的) CNNs require a fixed input image size (e.g., 224 224), which limits both the aspect ratio(宽高比) and the scale(尺度) of the input image. When applied to images of arbitrary(任意的) sizes, current methods mostly fit the input image to the fixed size, either via cropping [3], [4] or via warping (扭曲,使变形)[13], [7], as shown in Figure 1 (top). But the cropped region may not contain the entire object, while the warped content may result in unwanted geometric(几何学的) distortion(变形,失真). Recognition accuracy can be compromised(妥协,损害,危及) due to the content loss or distortion. Besides(除…之外,况且), a pre-defined scale may not be suitable when object scales vary. Fixing input sizes overlooks(忽略) the issues involving(涉及,包括,关于) scales.

&emsp; So why do CNNs require a fixed input size? A CNN mainly consists of two parts: convolutional layers, and fully-connected layers that follow. The convolutional layers operate in a sliding-window manner and output feature maps which represent the spatial arrangement of the activations (Figure 2). In fact, convolutional layers do not require a fixed image size and can generate feature maps of any sizes. On the other hand, the fully-connected layers need to have fixedsize/length input by their definition. Hence, the fixedsize constraint comes only from the fully-connected layers, which exist at a deeper stage of the network.

&emsp; In this paper, we introduce a spatial pyramid pooling (SPP) [14], [15] layer to remove the fixed-size constraint of the network. Specifically, we add an SPP layer on top of the last convolutional layer. The SPP layer pools the features and generates fixedlength outputs, which are then fed into the fullyconnected layers (or other classifiers). In other words, we perform some information “aggregation”(聚合,聚集) at a deeper stage of the network hierarchy(层级,层次结构,等级制度) (between convolutional layers and fully-connected layers) to avoid the need for cropping or warping at the beginning. Figure 1 (bottom) shows the change of the network architecture by introducing the SPP layer. We call the new network structure SPP-net.

&emsp; Spatial pyramid pooling [14], [15] (popularly known(俗称) as spatial pyramid matching or SPM [15]), as  an extension of the Bag-of-Words (BoW) model [16], is one of the most successful methods in computer vision. It partitions(划分,分开,磁盘分区) the image into divisions(分部;分类;分格;学部) from finer(更好的;更精细的;更锋利的) to coarser(粗糙的;粗俗的;更简略) levels, and aggregates local features(局部特征) in them. `SPP has long been(一直是) a key component in the leading and competition-winning systems for classification (e.g., [17], [18], [19]) and detection (e.g., [20]) before the recent prevalence of CNNs. (在CNN流行之前，SPP一直是classification和detection的 competition-winning systems 的关键组成部分.)` Nevertheless, SPP has not been considered in the context of CNNs. We note that SPP has several remarkable properties for deep CNNs: 1) SPP is able to generate a fixedlength output regardless of the input size, while the sliding window pooling used in the previous deep networks [3] cannot; 2) SPP uses multi-level spatial bins, while the sliding window pooling uses only a single window size. Multi-level pooling `has been shown(已然呈现)` to be robust to object deformations(变形) [15]; 3) `SPP can pool features extracted at variable scales thanks to(由于) the flexibility(灵活性) of input scales.(由于输入尺度的灵活性，SPP能够池化不同尺度下提取的features)` `Through(通过) experiments we show that all these factors elevate(提升,举起) the recognition accuracy of deep networks.(我们通过实验表明，这些因素都提高了深度网络的识别accuracy)`

&emsp; SPP-net not only makes it possible to generate representations from arbitrarily(任意地,武断地,反复无常地) sized images/windows for testing, but also allows us to feed images with varying sizes or scales during training. Training with variable-size images increases scale-invariance(尺度不变性) and reduces over-fitting. We develop a simple multi-size training method. `For a single network to accept variable input sizes, we approximate(近似,大概) it by multiple networks that share all parameters, while each of these networks is trained using a fixed input size. (对于接受可变输入大小的单个网络，我们通过共享所有参数的多个网络对其进行近似，而每个网络都使用固定的输入大小进行训练.)` In each epoch we train the network with a given input  size, and switch to another input size for the next epoch. Experiments show that this multi-size training converges `just as(正如,像…一样)` the traditional single-size training, and leads to better testing accuracy.

&emsp; The advantages of SPP are orthogonal to the specific CNN designs. In a series of `controlled experiments(对照试验,控制实验)` on the ImageNet 2012 dataset, `we demonstrate that SPP improves four different CNN architectures in existing publications [3], [4], [5] (or their modifications), over the no-SPP counterparts(职务相当的人,对应物,相当之物,副本). (我们证明了SPP在现有publications(或它们的modifications)中改进了四种不同的CNN架构，超过了no-SPP的同类publications.)` These architectures have various filter numbers/sizes, strides, depths, or other designs. It is thus reasonable for us to conjecture(推测,猜想) that SPP should improve more sophisticated (deeper and larger) convolutional architectures. SPP-net also shows state-of-the-art classification results on Caltech101 [21] and Pascal VOC 2007 [22] using only a single full-image representation and no fine-tuning.

SPP-net also shows great strength in object detection. In the leading object detection method R-CNN [7], the features from candidate windows are extracted via deep convolutional networks. This method shows remarkable detection accuracy on both the VOC and ImageNet datasets. But the feature computation in RCNN is time-consuming, because it repeatedly applies the deep convolutional networks to `the raw pixels of thousands of warped regions per image(每张图像的数千个扭曲区域的原始像素)`. In this paper, we show that we can run the convolutional layers only once on the entire image (regardless of the number of windows), and then extract features by SPP-net on the feature maps. This method yields a speedup of over one hundred times over R-CNN. Note that training/running a detector on the feature maps (rather than image regions) is actually a more popular idea [23], [24], [20], [5]. `But SPP-net inherits the power of the deep CNN feature maps and also the flexibility of SPP on arbitrary window sizes, which leads to outstanding accuracy and efficiency. (但是SPP-net继承了deep CNN feature maps的强大功能，也继承了SPP对于任意窗口大小的灵活性，这使得SPP-net具有很高的准确性和效率).` In our experiment, the SPP-net-based system (built upon the R-CNN pipeline) computes features 24-102 faster than R-CNN, while has better or comparable accuracy. With the recent fast proposal method of EdgeBoxes [25], our system takes 0.5 seconds processing an image (including all steps). This makes our method practical for real-world applications.

&emsp; A preliminary(初步的,开始的,预备的) version of this manuscript(手稿,原稿) has been published in ECCV 2014. Based on this work, we attended(参加,注意,照料,伴随) the competition of ILSVRC 2014 [26], and ranked #2 in object detection and #3 in image classification (`both are provided-data-only tracks(均为纯数据轨道)`) among all 38 teams. There are a few modifications made for ILSVRC 2014. We show that the SPP-nets can boost various networks that are deeper and larger (Sec. 3.1.2-3.1.4) over the no-SPP counterparts(职务相当的人,对应物,相当之物,副本). Further(更多的;更远的;进一步的;而且;此外), driven by our detection framework, `we find that multi-view testing on feature maps with flexibly located/sized windows (Sec. 3.1.5) can increase the classification accuracy. (我们发现在具有灵活位置/大小窗口的特征图上进行多视图测试可以提高分类accuracy).` This manuscript also provides the details of these modifications.

&emsp; We have released the code to facilitate(促进,帮助) future research (http://research.microsoft.com/en-us/um/people/kahe/).


# DEEP NETWORKS WITH SPATIAL PYRAMID POOLING

## Convolutional Layers and Feature Maps
&emsp; Consider the popular seven-layer architectures [3], [4]. The first five layers are convolutional, some of which are followed by pooling layers. These pooling layers can also be considered as convolutional , `in the sense(就…而言)` that they are using sliding windows. The last two layers are fully connected, with an N-way softmax as the output, where N is the number of categories.

&emsp; The deep network described above needs a fixed image size. `However, we notice that the requirement of fixed sizes is only due to the fully-connected layers that demand(需要,要求,查问) fixed-length vectors as inputs. (然而，我们注意到，固定大小的要求只是由于全连接层需要固定长度的向量作为输入).` On the other hand, the convolutional layers accept inputs of arbitrary sizes. The convolutional layers use sliding filters, and their outputs have `roughly(大致,粗糙地,粗略地) the same(大致相同的)` aspect ratio as the inputs. These outputs `are known as(被称为)` feature maps [1] - `they involve not only the strength of the responses, but also their spatial positions. (它们不仅涉及responses的强度，还涉及responses的空间位置).`

&emsp; In Figure 2, we visualize some feature maps. They are generated by some filters of the conv5 layer. Figure 2(c) shows the strongest activated images of these filters in the ImageNet dataset. `We see a filter can be activated by some semantic content. (我们看到filter能够被一些语义内容激活).` For example, the 55-th filter (Figure 2, bottom left) is most(大部分;最;非常;几乎) activated by a circle shape; the 66-th filter (Figure 2, top right) is most activated by $\land$-shape; and the 118-th filter (Figure 2, bottom right) is most activated by $\lor$-shape. These shapes in the input images (Figure 2(a)) activate the feature maps at the corresponding positions (the arrows in Figure 2).

&emsp; It is worth noticing that we generate the feature maps in Figure 2 without fixing the input size. These feature maps generated by deep convolutional layers are analogous(类似的) to the feature maps in traditional methods [27], [28]. In those methods, SIFT vectors [29] or image patches [28] are densely extracted and then encoded, e.g., by `vector quantization(矢量量化,矢量编码)` [16], [15], [30], sparse coding [17], [18], or Fisher kernels [19]. These encoded features consist of the feature maps, and are then pooled by Bag-of-Words (BoW) [16] or spatial pyramids [14], [15]. Analogously(类似地), the deep convolutional features can be pooled in a similar way.

## The Spatial Pyramid Pooling Layer
&emsp; The convolutional layers accept arbitrary input sizes,but they produce outputs of variable sizes. The classifiers (SVM/softmax) or fully-connected layers require fixed-length vectors. Such vectors can be generated by the Bag-of-Words (BoW) approach [16] that pools the features together. Spatial pyramid pooling [14], [15] improves BoW in that it can `maintain spatial information by pooling in local spatial bins. (通过池化局部空间bins来维持空间信息).` These spatial bins have sizes proportional(成比例的;相称的,均衡的) to the image size, so the number of bins is fixed regardless of the image size. This is in contrast to the sliding window pooling of the previous deep networks [3], where the number of sliding windows depends on the input size.

&emsp; To adopt the deep network for images of arbitrary sizes, we replace the last pooling layer (e.g., pool5, after the last convolutional layer) with a spatial pyramid pooling layer. Figure 3 illustrates our method. **In each spatial bin, we pool the responses of each filter (throughout this paper we use max pooling). The outputs of the spatial pyramid pooling are $kM$-dimensional vectors with the number of bins denoted as M (k is the number of filters in the last convolutional layer).** The fixed-dimensional vectors are the input to the fully-connected layer.

&emsp; With spatial pyramid pooling, the input image can  be of any sizes. This not only allows arbitrary aspect ratios, but also allows arbitrary scales. We can resize the input image to any scale (e.g., min(w, h)=180, 224, ...) and apply the same deep network. When the input image is at different scales, the network (with the same filter sizes) will extract features at different scales. The scales play important roles in traditional methods, e.g., the SIFT vectors are often extracted at multiple scales [29], [27] (determined(决定了的,坚决的) by the sizes of the patches and Gaussian filters). We will show that the scales are also important for the accuracy of deep networks.

&emsp; Interestingly, `the coarsest(粗糙的,粗俗的) pyramid level has a single bin that covers the entire image. (最粗的金字塔级别有一个单一的箱子，它覆盖了整个图像).` This is in fact a global pooling operation, `which is also investigated(investigate:研究,调查) in several concurrent works. (在几个并发工作中也对此进行了研究).` In [31], [32] a global average pooling is used to reduce the model size and also reduce overfitting; in [33], a global average pooling is used on the testing stage after all fc layers to improve accuracy; in [34], a global max pooling is used for weakly supervised object recognition. `The global pooling operation corresponds to the traditional Bag-of-Words method. (全局池化操作对应于传统的Bag-of-Words方法).`

-------------

主要思想已经讲完，其他的待续 ...

## Training the Network




