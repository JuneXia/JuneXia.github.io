---
title: 
date: 2020-03-26
tags:
categories: ["深度学习笔记"]
mathjax: true
---
SSD: Single Shot MultiBox Detector \
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
Scott Reed, Cheng-Yang Fu, Alexander C. Berg \
UNC Chapel Hill, Zoox Inc. Google Inc. University of Michigan, Ann-Arbor
<!-- more -->

**Abstract**. We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, `discretizes(discretize: vt.使离散;离散化) the output space of bounding boxes into a set of default boxes` **over** `different aspect ratios and scales per feature map location.(将bounding boxes的输出离散化为一组default boxes，`**根据**`每个feature map location的aspect ratios和scales来离散化).` 
At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. 
在预测阶段，我们的网络为每个default box的目标分类生成一个分数，并且产生调整box以更好地匹配目标形状。
Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. 
结合多个不同分辨率feature map的预测来处理不同尺寸的目标。
SSD is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent(后来的;随后的;后面的) pixel or feature resampling stages and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, COCO, and ILSVRC datasets confirm that SSD has competitive accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. For 300 × 300 input, SSD achieves 74.3% mAP1 on VOC2007 test at 59 FPS on a Nvidia Titan X and for 512 × 512 input, SSD achieves 76.9% mAP, outperforming a comparable state-of-the-art Faster R-CNN model. Compared to other single stage methods, SSD has much better accuracy even with a smaller input image size. Code is available at: https://github.com/weiliu89/caffe/tree/ssd . 

Keywords: Real-time Object Detection; Convolutional Neural Network

# Introduction
&emsp; Current state-of-the-art object detection systems are variants(n.[计] 变体;变异型) of the following approach: hypothesize(v. 假设,假定) bounding boxes, resample pixels or features for each box, and apply a highquality classifier. This pipeline has prevailed on detection benchmarks since the Selective Search work [1] through the current leading results on PASCAL VOC, COCO, and ILSVRC detection all based on Faster R-CNN[2] albeit(conj. 虽然,尽管) with deeper features such as [3]. While accurate, these approaches have been too computationally intensive(加强的;集中的;透彻的;加强语气的) for embedded systems and, even with high-end hardware, too slow for real-time applications.
尽管当前最先进的Faster R-CNN检测精度很高，但其速度在嵌入式等要求实时应用的场合还是很慢。

Often detection speed for these approaches is measured in seconds per frame (SPF), and even the fastest high-accuracy detector, Faster R-CNN, operates at only 7 frames per second (FPS). There have been many attempts to build faster detectors by attacking each stage of the detection pipeline (see related work in Sec. 4), but so far, significantly increased speed comes only at the cost of significantly decreased detection accuracy.
这些方法的检测速度通常是以秒来进行评测的，即使是具有高准确率的Faster R-CNN也才只有7FPS。尽管已经有许多尝试通过攻击detection pipeline的各个阶段来构建更快的detector，但是迄今为止，显著提升速度的方法只有显著降低检测准确率。


&emsp; This paper presents the first deep network based object detector that does not `resample pixels or features for bounding box hypotheses` and and is as accurate as approaches that do. 
> `bounding box hypotheses`就是Region Proposal，为Region Proposal所做的`resample pixels or features`实际上就是RPN网络。
所以上句话意思就是说：本文率先提出了一种基于目标检测器的深度网络，该方法不需要RPN，并且和需要RPN的方法拥有同样的准确率。

This results in a significant improvement in speed for high-accuracy detection (59 FPS with mAP 74.3% on VOC2007 test, vs. Faster R-CNN 7 FPS with mAP 73.2% or YOLO 45 FPS with mAP 63.4%). The fundamental improvement in speed comes from eliminating bounding box proposals and the subsequent pixel or feature resampling stage. We are not the first to do this (cf [4,5]), but by adding a series of improvements, we manage to increase the accuracy significantly over previous attempts. 
Our improvements include using a small convolutional filter to predict object categories and offsets in bounding box locations, &emsp;&emsp;&emsp;&emsp; using **separate predictors (filters)** for different aspect ratio detections, and applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple scales. 
> what is **separate predictors (filters)** ? 待后续确认

With these modifications—especially using multiple layers for prediction at different scales—we can achieve high-accuracy using relatively low resolution input, further increasing detection speed. 
`While these contributions may seem small independently(虽然这些贡献单独看起来略显渺小)`, we note that the resulting system improves accuracy on real-time detection for PASCAL VOC from 63.4% mAP for YOLO to 74.3% mAP for our SSD. 
This is a larger relative improvement in detection accuracy than that from the recent, very `high-profile(高调的;备受瞩目的;知名度高的)` work on residual networks [3]. Furthermore, significantly improving the speed of high-quality detection can broaden the range of settings where computer vision is useful.

We summarize our contributions as follows: 
- We introduce SSD, a single-shot detector for multiple categories that is faster than the previous state-of-the-art for single shot detectors (YOLO), and significantly more accurate, in fact as accurate as slower techniques that perform explicit region proposals and pooling (including Faster R-CNN).
- The core of SSD is predicting category scores and box offsets for a fixed set of default bounding boxes using small convolutional filters applied to feature maps. 
在feature maps上使用小卷积filters来预测一组default bounding boxes的类别分数和box偏移量。
- To achieve high detection accuracy we produce predictions of different scales from feature maps of different scales, and explicitly separate predictions by aspect ratio. 
为了实现较高的检测精度，我们从不同尺度的特征图中生成不同尺度的预测，并通过纵横比明确地分离预测。
- These design features lead to simple end-to-end training and high accuracy, even on low resolution input images, further improving the speed vs accuracy trade-off(n. 交换,交易;权衡;协定). 
- Experiments include timing and accuracy analysis on models with varying input size evaluated on PASCAL VOC, COCO, and ILSVRC and are compared to `a range of(一系列,一些,一套)` recent state-of-the-art approaches.


# The Single Shot Detector (SSD)
&emsp; This section describes our proposed SSD framework for detection (Sec. 2.1) and the associated training methodology (Sec. 2.2). Afterwards, Sec. 3 presents dataset-specific model details and experimental results.

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/SSD1.jpg" width = 60% height = 60% />
</div>
> Fig. 1: SSD framework. (a) SSD only needs an input image and ground truth boxes for each object during training. In a convolutional fashion, we evaluate a small set (e.g. 4) of default boxes of different aspect ratios at each location in several feature maps with different scales (e.g. 8 × 8 and 4 × 4 in (b) and \(c\)). 
For each default box, we predict both the shape offsets and the confidences for all object categories ((c1, c2, · · · , cp)). At training time, we first match these default boxes to the ground truth boxes. 
For example, we have matched two default boxes with the cat and one with the dog, which are treated as positives and the rest as negatives. 
The model loss is a weighted sum between localization loss (e.g. Smooth L1 [6]) and confidence loss (e.g. Softmax).

## Model
&emsp; The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, SSD是一个基于前馈卷积网络的方法，它产生一个固定尺寸的bounding boxes集合，并对这些boxes中存在的目标类别实例进行打分。
followed by a non-maximum suppression step to produce the final detections. 然后执行一个非极大值抑制步骤以产生最后的检测结果。
The early network layers are based on a standard architecture used for high quality image classification (truncated before any classification layers), which we will call the base network.
> 作者是使用VGG作为这个base network的，当然也可以使用其他网络结构

We then add auxiliary structure to the network to produce detections with the following key features:
**Multi-scale feature maps for detection** We add convolutional feature layers to the end of the truncated base network. These layers decrease in size progressively(渐进地;逐步的;日益增多地) and allow predictions of detections at multiple scales. 
The convolutional model &emsp; for predicting detections &emsp; is different for each feature layer (cf Overfeat[4] and YOLO[5] that operate on a single scale feature map). 
每个特征层用于预测检测的卷积模型是不同的。
**Convolutional predictors for detection** Each added feature layer (or optionally an existing feature layer from the base network) can produce a fixed set of detection predictions using a set of convolutional filters. 每个添加的feature layer使用一组卷积filters进行预测并产生一组固定的检测结果。
These are indicated on top of the SSD network architecture in Fig. 2. 
For a feature layer of size m × n with p channels, the basic element for predicting parameters of a potential detection is a 3 × 3 × p small kernel that produces either a score for a category, or a shape offset relative to the default box coordinates. 
> 蹩脚直译：
> 对于一个有 $p$ 个channel的m × n大小的feature layer，用来预测~~一个潜在检测的参数~~的基本元素是一个 3 × 3 × p 的小kernel，它既可以生成一个类别分数，也可以生成一个相对default box坐标的偏移。
> 
> 概括翻译：
> 对于一个有 p 个channel的m × n大小的feature layer，使用 3 × 3 × p 的小kernel进行预测操作，它既可以生成类别分数也可以生成一个相对default box坐标的偏移。

At each of the m × n locations where the kernel is applied, it produces an output value. `The bounding box offset output values are measured relative to a default box position relative to each feature map location (bounding box偏移量输出值相对于每个feature map位置的default box位置进行测量)` (*cf* the architecture of YOLO[5] that uses an intermediate(n. adj. 中间(的),过渡(的)) fully connected layer instead of a convolutional filter for this step). 

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/SSD2.jpg" width = 60% height = 60% />
</div>
> Fig. 2: A comparison between two single shot detection models: SSD and YOLO [5]. Our SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences. SSD with a 300 × 300 input size significantly outperforms(vt. 胜过;做得比……好) its 448 × 448 YOLO counterpart in accuracy on VOC2007 test while also improving the speed.

**Default boxes and aspect ratios** We associate a set of default bounding boxes with each feature map cell, for multiple feature maps at the top of the network. 
我们将一组default bounding boxes与每个feature map cell关联起来，用于网络顶部的多个feature maps。
The default boxes tile the feature map in a convolutional manner, so that the position of each box relative to its corresponding cell is fixed. 
default boxes以卷积方式平铺feature map，因此每个box相对于其相应cell的位置是固定的。
At each feature map cell, we predict the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. 
在每个feature map cell中，我们预测相对于cell中的default box形状的偏移量，以及每个类的分数，这表示每个boxes中存在一个类实例。
Specifically, for each box out of k at a given location, we compute c class scores and the 4 offsets relative to the original default box shape. 
具体地说，对于给定位置上k之外的每个方块，我们计算c个类别的分数和相对于原始default box形状的4个偏移量。
This results in a total of $(c + 4)k$ filters that are applied around each location in the feature map, yielding $(c + 4)kmn$ outputs for a $m × n$ feature map. For an illustration of default boxes, please refer to Fig. 1. Our default boxes are similar to the anchor boxes used in Faster R-CNN [2], however we apply them to several feature maps of different resolutions. 
Allowing different default box shapes in several feature maps let us efficiently discretize(v. 离散) the space of possible output box shapes.
允许在多个feature maps中使用不同的default box形状 使得我们可以有效地discretize可能的输出box形状空间。

## Training
&emsp; The key difference between training SSD and training a typical detector that uses region proposals, is that ground truth information needs to be assigned to specific outputs in the fixed set of detector outputs. 
训练SSD和其他使用region proposals的典型检测器的不同之处关键在于 ground truth信息需要被分配到一组固定检测器的指定输出。
Some version of this is also required for training in YOLO[5] and for the region proposal stage of Faster R-CNN[2] and MultiBox[7]. Once this assignment is determined, the loss function and back propagation are applied end-to-end. 
Training also involves choosing the set of default boxes and scales for detection as well as the hard negative mining and data augmentation strategies.
训练还包括为检测器选择一组default boxes和scales，以及难例挖掘和数据增强策略。


**Matching strategy** During training we need to determine which default boxes correspond to a ground truth detection and train the network accordingly(adv. 因此;于是;相应地;照着). 
在训练过程中，我们需要确定哪些default boxes对应于ground truth，并相应地训练网络。
For each ground truth box we are selecting from default boxes that vary over location, aspect ratio, and scale. 
对于每个ground truth框，我们选择的是根据位置、长宽比和比例而变化的默认框。
We begin by matching each ground truth box to the default box with the best jaccard overlap (as in MultiBox [7]). Unlike MultiBox, we then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5). This simplifies the learning problem, allowing the network to predict high scores for multiple overlapping default boxes rather than requiring it to pick(选择,挑选;采摘;挖) only the one with maximum overlap. 

**Training objective** The SSD training objective is derived from the MultiBox objective [7,8] but is extended to handle multiple object categories. 
Let $x^p_{ij} = \{1, 0\}$ be an indicator for matching the $i$-th default box to the $j$-th ground truth box of category p. In the matching strategy above, we can have $\sum_i x^p_{ij} \geq 1$. 
假设 $x^p_{ij} = \{1, 0\}$ 是匹配第 i 个default box和第 j 个类别为 p 的 ground truth box 的目标。
> 简单来说，就是如果 default box 和 ground truth box 匹配，则 $x^p_{ij} = 1$，否则 $x^p_{ij} = 0$。可以理解为《信号与系统》中的冲激函数或抽样函数。

The overall objective loss function is a weighted sum of the localization loss (loc) and the confidence loss (conf):
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/SSD3.jpg" width = 60% height = 60% />
</div>

where $N$ is the number of matched default boxes. If $N = 0$, wet set the loss to 0. The localization loss is a Smooth L1 loss [6] between the predicted box (l) and the ground truth box (g) `parameters`.
> 这里的`parameters`不知为何意，TODO.

Similar to Faster R-CNN [2], we regress to offsets for the center ($cx, cy$) of the default bounding box ($d$) and for its width ($w$) and height ($h$).
与Faster R-CNN类似，我们回归与default bounding box的中心($cx, cy$)以及宽($w$)和高($h$)的偏移量。
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/SSD4.jpg" width = 60% height = 60% />
</div>

> 我的一些理解：
> $\hat{g}^m_j$ 可以理解为ground truth相对于default box的偏移量，$l^m_i$是predicted box相对于default box的偏移量，而 $L1(l^m_i - \hat{g}^m_j)$ 就是希望 $l^m_i$与$\hat{g}^m_j$ 之间越小越好。
> 
> 那么怎么来衡量这个偏移量呢？
> 对于中心点($cx, cy$)偏移量的衡量，$cx$ 是通过公式 $(g^{cx}_j - d^{cx}_i) / d^w_i$ 来衡量的，$cy$ 类似；
> 而对于宽$w$和高$h$的偏移量，$w$ 是通过公式 $\text{log}(\frac{g^w_j}{d^w_i})$ 来进行衡量的，$h$ 类似。
> 为什么这样做，可参考文献 [m1]

The confidence loss is the softmax loss over multiple classes confidences ($c$)
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/SSD5.jpg" width = 60% height = 60% />
</div>
and the weight term $\alpha$ is set to 1 by cross validation.


**Choosing scales and aspect ratios for default boxes** To handle different object scales, some methods [4,9] suggest processing the image at different sizes and combining the results afterwards(adv. 后来;然后). However, by utilizing feature maps from several different layers in a single network for prediction we can mimic(vt. 模仿,摹拟. adj. 模仿的,模拟的;假装的) the same effect, while also sharing parameters across all object scales. Previous works [10,11] have shown that using feature maps from the lower layers can improve semantic segmentation quality because the lower layers capture more fine details of the input objects. Similarly, [12] showed that adding global context pooled from a feature map can help smooth the segmentation results.

`motivate`: v. 刺激，使有动机，激发…的积极性；成为……的动机；给出理由；申请 \
`Motivated` by these methods, we use both the lower and upper feature maps for detection. Figure 1 shows two exemplar feature maps (8×8 and 4×4) which are used in the framework. 
In practice, we can use many more with small computational `overhead(n. 经常性支出，运营费用（常用复数 overheads）；（飞机的）顶舱；adj. 在头上方的，在空中的；地面以上的，高架的；（费用、开支等）经常的，日常的；adv. 在头顶上方，在空中；在高处). `
在实践中，我们可以用较小的计算开销使用更多的feature maps。

&emsp; Feature maps from different levels within a network are known to have different ( empirical(经验主义的) ) receptive field sizes [13]. Fortunately(幸运地), within the SSD framework, the default boxes do not necessary need to correspond to the actual receptive fields of each layer. 
We design the tiling of default boxes so that specific feature maps learn to be responsive to particular(adj. 特定的,特别的;详细的;独有的;挑剔的) scales of the objects. 
我将default boxes设计成平铺形式，以至于指定的feature maps能够学习到特定目标尺度的响应。
Suppose we want to use $m$ feature maps for prediction. The scale of the default boxes for each feature map is computed as:
$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m - 1} (k - 1), \qquad k \in [1, m]  \tag{4}$$

where $s_{min}$ is $0.2$ and $s_{max}$ is $0.9$, meaning the lowest layer has a scale of $0.2$ and the highest layer has a scale of $0.9$, and all layers in between are regularly(adv. 定期地；有规律地；整齐地；匀称地) spaced. 
在这之间所有的layers都是有规律的间隔着。\
We impose different aspect ratios for the default boxes, and denote them as $a_r \in \{1, 2, 3, \frac{1} {2} , \frac{1} {3} \}$. We can compute the width ($w^a_k = s_k \sqrt{a_r}$) and height ($h^a_k = s_k / \sqrt{a_r}$) for each default box. 
> 根据 $a_r$ 的取值，default box 的宽和高有下面几种组合：

$$
\begin{aligned}
\text{while} \ \ & a_r = 1, \ w^a_k = s_k, \ h^a_k = s_k \\
\text{while} \ \ & a_r = 2, \ w^a_k = s_k \sqrt{2}, \ h^a_k = \frac{s_k} {\sqrt{2} } \\
\text{while} \ \ & a_r = 3, \ w^a_k = s_k \sqrt{3}, \ h^a_k = \frac{s_k} {\sqrt{3} } \\
\text{while} \ \ & a_r = \frac{1} {2}, \ w^a_k = \frac{s_k} {\sqrt{2} }, \ h^a_k = s_k \sqrt{2} \\
\text{while} \ \ & a_r = \frac{1} {3}, \ w^a_k = \frac{s_k} {\sqrt{3} }, \ h^a_k = s_k \sqrt{3} \\
\end{aligned}
$$

For the aspect ratio of 1, we also add a default box whose scale is $s'_k = \sqrt{s_k s_{k+1}}$, resulting in 6 default boxes per feature map location. 
> 对于 $a_r = 1$ 的时候，我们还增加了一种 default box，该 default box 的尺度是 $s'_k = \sqrt{s_k s_{k+1}}$，综上，每个feature map位置中一共有6种 default box。

We set the center of each default box to $(\frac{i + 0.5} {\vert f_k \vert}, \frac{j + 0.5} {\vert f_k \vert})$, where $\vert f_k \vert$ is the size of the $k$-th square feature map, $i, j \in [0, |fk|$). In practice, one can also design a distribution of default boxes to best fit a specific dataset. How to design the optimal tiling is an open question as well.

&emsp; By combining predictions for all default boxes with different scales and aspect ratios from all locations of many feature maps, we have a diverse set of predictions, covering various input object sizes and shapes. 
通过联合预测多个feature maps中 ~~所有位置的~~ 不同尺度和长宽比的default boxes，我们可以得到多种预测组合，能够覆盖多种尺寸和形状的输入目标。
For example, in Fig. 1, the dog is matched to a default box in the 4 × 4 feature map, but not to any default boxes in the 8 × 8 feature map. This is because those boxes have different scales and do not match the dog box, and therefore are considered as negatives during training.

基本思想就到这里，其他的待续。。。

**Hard negative mining** ……

**Data augmentation** ……


# Experimental Results

## PASCAL VOC2007

## Model analysis

## PASCAL VOC2012

## COCO

## Preliminary ILSVRC results

## Data Augmentation for Small Object Accuracy

## Inference time

# Related Work

# Conclusions

# Acknowledgment

# References
1. Uijlings, J.R., van de Sande, K.E., Gevers, T., Smeulders, A.W.: Selective search for object recognition. IJCV (2013) 
2. Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. In: NIPS. (2015) 
3. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. (2016) 
4. Sermanet, P., Eigen, D., Zhang, X., Mathieu, M., Fergus, R., LeCun, Y.: Overfeat: Integrated recognition, localization and detection using convolutional networks. In: ICLR. 
5. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: CVPR. (2016)


[m1] [FasterRcnn中boundingbox regression的一些理解](https://blog.csdn.net/qian99/article/details/82218963)


