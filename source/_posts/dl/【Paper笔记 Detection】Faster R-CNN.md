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


**Abstract**
&emsp; State-of-the-art object detection networks depend on region proposal algorithms to hypothesize(假设,假定) object locations.  Advances(发展,前进,提出) like SPPnet [1] and Fast R-CNN [2] have reduced the running time of these detection networks, exposing(expose,遗弃,陈列,揭露) region  proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image  convolutional features with the detection network, thus enabling nearly(几乎) cost-free region proposals. An RPN is a fully convolutional  network that simultaneously(同时地) predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to  generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN  into a single network by sharing their convolutional features using the recently popular terminology(专业术语,用辞) of neural networks with “attention” mechanisms(注意力机制), the RPN component tells the unified(unify:统一,使一致) network where to look. For the very deep VGG-16 model [3],  our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection  accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO  2015 competitions, Faster R-CNN and RPN are the foundations(基础,房基) of the 1st-place winning entries(entry:进入,条目,记录) in several tracks. Code has been  made publicly available.


# Introduction
&emsp; Recent advances(advance:n.发展,前进;v.提出,使…前进) in object detection are driven by the success of region proposal methods (e.g., [4]) and region-based convolutional neural networks (RCNNs) [5]. `Although region-based CNNs were computationally expensive as originally developed in [5], their cost has been drastically(彻底地,激烈地,大幅度地) reduced thanks to sharing convolutions across proposals [1], [2]. (虽然基于区域的CNNs的计算成本与最初在[5]中开发时一样高，但是由于proposals之间共享卷积，它们的成本已经大大降低了).` The latest incarnation(化身,典型), Fast R-CNN [2], achieves near real-time rates using very deep networks [3], when ignoring the time spent on region proposals. Now, proposals are the test-time computational bottleneck in state-of-the-art detection systems.

&emsp; Region proposal methods typically(典型地,代表性地) rely on `inexpensive features and economical inference schemes(廉价的特征和经济的推理方案)`. Selective Search [4], one of the most popular methods, `greedily(贪婪地) merges superpixels(超像素) based on engineered low-level features.(它贪婪地合并基于工程底层特征的超像素).` Yet when compared to efficient detection networks [2], `Selective Search is an order of magnitude slower, at 2 seconds per image in a CPU implementation. (选择性搜索要慢了一个数量级，在CPU实现中，每幅图像要慢2秒).` EdgeBoxes [6] currently provides the best tradeoff(权衡,折中,(公平)交易) between proposal quality and speed, at 0.2 seconds per image. Nevertheless, the region proposal step still consumes as much running time as the detection network.

&emsp; `One may note that(人们可能会注意到)` fast region-based CNNs take advantage of GPUs, `while(在…期间;与…同时;(比对两件事物)然而;虽然,尽管;直到…为止;adv.在…时候) the region proposal methods used in research are implemented on the CPU(而研究中使用的区域建议方法是在CPU上实现的)`, making such runtime comparisons inequitable(不公平的). An obvious(明显的,显著的) way to accelerate proposal computation is to reimplement it for the GPU. This may be an effective engineering solution, but re-implementation ignores the down-stream(下游) detection network and therefore misses important opportunities for sharing computation.


# Related Work

# Faster R-CNN
&emsp; Our object detection system, called Faster R-CNN, is  composed of two modules. The first module is a deep  fully convolutional network that proposes regions,  and the second module is the Fast R-CNN detector [2]  that uses the proposed regions. The entire system is a single, unified network for object detection (Figure 2). Using the recently popular terminology(专业术语,用辞) of neural networks with attention [31] mechanisms, the RPN module tells the Fast R-CNN module where to look. In Section 3.1 we introduce the designs and properties of the network for region proposal. In Section 3.2 we develop algorithms for training both modules with features shared. 

## Region Proposal Networks
&emsp; A Region Proposal Network (RPN) takes an image  (of any size) as input and outputs a set of rectangular  object proposals, each with an objectness score. `We  model(模型,典范,模拟) this process with a fully convolutional network  [7], which we describe in this section. (我们用一个全卷积网络来模拟这个过程，我们将在本节中描述它).` Because our ultimate goal is to share computation with a Fast R-CNN  object detection network [2], we assume that both nets  share a common set of convolutional layers. In our experiments, we investigate(调查,研究) the Zeiler and Fergus model  [32] (ZF), which has 5 shareable convolutional layers  and the Simonyan and Zisserman model [3] (VGG-16),  which has 13 shareable convolutional layers.

&emsp; To generate region proposals, `we slide a small network over the convolutional feature map output by the last shared convolutional layer. (我们在最后一个共享卷积层的卷积特征图输出上滑动一个小网络)` This small network `takes as input(take…as input,以…为输入) an $n \times n$ spatial window of the input convolutional feature map. (以输入卷积特征图的一个n×n空间窗口作为输入).` Each sliding window is mapped(map: n.地图,示意图;v.绘制地图,映射) to a lower-dimensional feature (256-d for ZF and 512-d for VGG, with ReLU [33] following). This feature is fed into two sibling(兄弟姐妹,同级的) fullyconnected layers a box-regression layer (reg) and a box-classification layer (cls). We use n = 3 in this paper, noting that the effective receptive field on the input image is large (171 and 228 pixels for ZF and VGG, respectively). `This mini-network is illustrated at a single position in Figure 3 (left). (图3(左)显示了这个迷你网络的一个位置).` `Note that because the mini-network operates in a sliding-window fashion, the fully-connected layers are shared across(从…的一边到另一边,穿过,越过;在对面,横过) all spatial locations. (请注意，由于微型网络以滑动窗口的方式运行，因此所有空间位置都共享完全连接的层).` `This architecture is naturally implemented with an n n convolutional layer followed by two sibling 1 1 convolutional layers (for reg and cls, respectively). (这个架构很自然地通过一个n×n卷积层和两个同级的1×1卷积层(分别用于reg和cls)来实现).`

### Anchors
&emsp; At each sliding-window location, we simultaneously(同时地) predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as $k$. So the reg layer has $4k$ outputs encoding the coordinates of $k$ boxes, and the $cls$ layer outputs $2k$ scores that estimate probability of object or not object for each proposal. `The k proposals are parameterized relative to k reference boxes, which we call anchors. (k个proposal是相对于k个参考框参数化的，我们称之为锚).` `An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio (Figure 3, left). (锚位于所讨论的滑动窗口的中心，并与比例和高宽比相关联(图3,左))` By default we use 3 scales and 3 aspect ratios, yielding $k = 9$ anchors at each sliding position. For a convolutional feature map of a size $W \times H$ (typically ~2,400), there are $W \times H \times k$ anchors in total.

**Translation-Invariant Anchors**(平移不变锚)
&emsp; An important property of our approach is that it  is translation invariant, `both in terms of the anchors  and the functions that compute proposals relative to(相对于,涉及) the anchors. (无论是在锚点方面，还是在计算相对于锚点的proposal functon方面)` `If one translates an object in an image,  the proposal should translate and the same function should be able to predict the proposal in either location. (如果某个时候移动了图像中的一个目标，那么他的proposal也应该被平移，同样的function应该能够预测任何位置的proposal)` This translation-invariant property is guaranteed by our method5. As a comparison, the MultiBox  method [27] uses k-means to generate 800 anchors,  which are not translation invariant. So MultiBox does  not guarantee that the same proposal is generated if  an object is translated.

&emsp; The translation-invariant property also reduces the  model size. MultiBox has a $(4 + 1) \times 800$-dimensional  fully-connected output layer, whereas our method has  a $(4 + 2) \times 9$-dimensional convolutional output layer  in the case of k = 9 anchors. As a result, our output  layer has $2.8 \times 10^4$ parameters $(512 \times (4 + 2) \times 9$ for VGG-16), `two orders of magnitude fewer than  MultiBox's output layer that has $6.1 \times 10^6$ parameters $(1536 \times (4 + 1) \times 800$ for GoogleNet [34] in MultiBox  [27]). (比MultiBox的输出层少两个数量级，后者有`$6.1 \times 10^6$`个参数).` If considering the feature projection(投影,规划) layers, our  proposal layers still have an order of magnitude fewer  parameters than MultiBox[^1]. We expect our method  to have less risk of overfitting on small datasets, like  PASCAL VOC.

> [^1]: Considering the feature projection layers, our proposal layers’ parameter count is 3 × 3 × 512 × 512 + 512 × 6 × 9 = 2:4 × 106; MultiBox’s proposal layers’ parameter count is 7 × 7 × (64 + 96 + 64 + 64) × 1536 + 1536 × 5 × 800 = 27 × 106.


**Multi-Scale Anchors as Regression References**
&emsp; `Our design of anchors presents(我们的anchors设计提出了)` a novel scheme  for addressing multiple scales (and aspect ratios). As  shown in Figure 1, there have been two popular ways  for multi-scale predictions. The first way is based on  image/feature pyramids, e.g., in DPM [8] and CNNbased methods [9], [1], [2]. The images are resized at  multiple scales, and feature maps (HOG [8] or deep  convolutional features [9], [1], [2]) are computed for  each scale (Figure 1(a)). This way is often useful but  is time-consuming. The second way is to use sliding  windows of multiple scales (and/or aspect ratios) on  the feature maps. For example, in DPM [8], models  of different aspect ratios are trained separately(分别地;分离地;个别地) using  different filter sizes (such as $5 \times 7$ and $7 \times 5$). If this way  is used to address multiple scales, `it can be thought(think:n.思考;想法;关心;v.想,思考;认为) of as a  pyramid of filters(它可以被认为是一个“过滤器的金字塔”)` (Figure 1(b)). The second  way is usually adopted jointly with the first way [8].

&emsp; As a comparison, our anchor-based method is built on a pyramid of anchors, which is more cost-efficient(经济有效的). Our method classifies and regresses bounding boxes with reference to anchor boxes of multiple scales and aspect ratios. It only relies on images and feature maps of a single scale, and uses filters (sliding windows on the feature map) of a single size. We show by experiments the effects of this scheme for addressing multiple scales and sizes (Table 8).

&emsp; Because of this multi-scale design based on anchors, we can simply use the convolutional features computed on a single-scale image, as is also done by the Fast R-CNN detector [2]. `The design of multiscale anchors is a key component for sharing features without extra cost for addressing scales. (多尺度anchors的设计是实现特征共享的关键环节，而不需要额外的处理尺度成本).`

### Loss Function
For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors: (i) the anchor/anchors with the highest Intersection-overUnion (IoU) overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. Note that a single ground-truth box may assign positive labels to multiple anchors. Usually the second condition is sufficient to determine the positive samples; but we still adopt the first condition for the reason that in some rare cases the second condition may find no positive sample. We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. Anchors that are neither positive nor negative do not contribute to the training objective. 

&emsp; With these definitions, we minimize an objective function following the multi-task loss in Fast R-CNN [2]. Our loss function for an image is defined as:
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/FasterRCNN-loss1.jpg" width = 80% height = 80% />
</div>

Here, $i$ is the index of an anchor in a mini-batch and $p_i$ is the predicted probability of anchor $i$ being an object. The ground-truth label $p_i^*$ is 1 if the anchor is positive, and is 0 if the anchor is negative. $t_i$ is a vector representing the 4 parameterized coordinates of the predicted bounding box, and $t_i^*$ is that of the ground-truth box associated with a positive anchor. The classification loss $L_{cls}$ is log loss over two classes (object $vs$. not object). For the regression loss, we use $L_{reg}(t_i, t_i^*) = R(t_i - t_i^*)$ where $R$ is the robust loss function (smooth L1) defined in [2]. The term $p_i^* L_{reg}$ means the regression loss is activated only for positive anchors ($p_i^* = 1)$ and is disabled otherwise ($p_i^* = 0$). The outputs of the $cls$ and $reg$ layers consist of ${p_i}$ and ${t_i}$ respectively.

&emsp; The two terms are normalized by $N_{cls}$ and $N_{reg}$ and weighted by a balancing parameter $λ$. In our current implementation (as in the released code), the $cls$ term in Eqn.(1) is normalized by the mini-batch size (i.e., $N_{cls} = 256$) and the $reg$ term is normalized by the number of anchor locations (i.e., $N_{reg} \sim 2400$). By default we set $λ = 10$, and thus both $cls$ and $reg$ terms are roughly equally weighted. We show by experiments that the results are insensitive to the values of $λ$ in a wide range (Table 9). We also note that the normalization as above is not required and could be simplified.

&emsp; For bounding box regression, we adopt the parameterizations of the 4 coordinates following [5]:
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/FasterRCNN-loss2.jpg" width = 80% height = 80% />
</div>

where $x, y, w$, and $h$ denote the box's center coordinates and its width and height. Variables $x, x_a$, and $x$ are for the predicted box, anchor box, and groundtruth box respectively (likewise for $y, w, h$). This can be thought of as bounding-box regression from an anchor box to a nearby ground-truth box.

> 关于公示(2)的一些理解可以参考文献 [1]

&emsp; Nevertheless, our method achieves bounding-box regression by a different manner(n.方式;习惯;种类;规矩;风俗) from previous RoIbased (Region of Interest) methods [1], [2]. In [1], [2], bounding-box regression is performed on features pooled from *arbitrarily*(adv.任意地;反复无常地;专横地) sized RoIs, and the regression weights are *shared* by all region sizes. In our formulation, the features used for regression are of the same spatial size (3 × 3) on the feature maps. To account for varying sizes, a set of k bounding-box regressors(n.[数]回归量) are learned. Each regressor is responsible for one scale and one aspect ratio, and the k regressors do not share weights. As such, it is still possible to predict boxes of various sizes even though the features are of a fixed size/scale, thanks to the design of anchors.

### Training RPNs
The RPN can be trained end-to-end by backpropagation and stochastic gradient descent (SGD) [35]. We follow the image-centric sampling strategy from [2] to train this network. **Each mini-batch `arises from(来自)` a single image that contains many positive and negative example anchors.** It is possible to optimize for the loss functions of all anchors, but this will bias towards negative samples as they are dominate. Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1. If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones.

&emsp; We randomly initialize all new layers by drawing weights from a zero-mean Gaussian distribution with standard deviation 0.01. All other layers (i.e., the shared convolutional layers) are initialized by pretraining a model for ImageNet classification [36], as is standard practice [5]. We tune all layers of the ZF net, and conv3 1 and up for the VGG net to conserve memory [2]. We use a learning rate of 0.001 for 60k mini-batches, and 0.0001 for the next 20k mini-batches on the PASCAL VOC dataset. We use a momentum of 0.9 and a weight decay of 0.0005 [37]. Our implementation uses Caffe [38].

## Sharing Features for RPN and Fast R-CNN
`Thus far(迄今为止)` we have described how to train a network for region proposal generation, without considering the region-based object detection CNN that will utilize these proposals. For the detection network, we adopt Fast R-CNN [2]. Next we describe algorithms that learn a unified network composed of RPN and Fast R-CNN with shared convolutional layers (Figure 2).

&emsp; Both RPN and Fast R-CNN, trained independently, will modify their convolutional layers in different ways. We therefore need to develop a technique that **allows for sharing convolutional layers between the two networks, rather than learning two separate networks**. We discuss three ways for training networks with features shared:

(i) *Alternating(adj.交替的;交互的) training*. In this solution, we first train RPN, and use the proposals to train Fast R-CNN. The network tuned by Fast R-CNN is then used to initialize RPN, and this process is iterated(iterate vt.迭代;重复). This is the solution that is used in all experiments in this paper.

(ii) *Approximate(n/adj.近似(的),大概(的)) joint training*. In this solution, the RPN and Fast R-CNN networks are merged into one network during training as in Figure 2. In each SGD iteration, the forward pass generates region proposals which are treated just like fixed, pre-computed proposals when training a Fast R-CNN detector. The backward propagation `takes place as usual(一切如常,照常进行)`, where for the shared layers the backward propagated signals from both the RPN loss and the Fast R-CNN loss are combined. This solution is easy to implement. But this solution ignores the derivative(n.派生物;金融衍生产品;派生词;(化学)衍生物;导数) w.r.t. the proposal boxes coordinates that are also network responses, so is approximate. In our experiments, we have empirically(adv.以经验为主地) found this solver produces close(接近的) results, yet reduces the training time by about 25-50% comparing with alternating training. This solver is included in our released Python code.

未完待续。。。

# 参考文献
[1] [关于RPN中proposal的坐标回归参数的一点理解及Faster R-CNN的学习资料](https://blog.csdn.net/Blateyang/article/details/84800007)



