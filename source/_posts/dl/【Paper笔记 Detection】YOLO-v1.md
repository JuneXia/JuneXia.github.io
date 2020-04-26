---
title: 
date: 2020-04-22
tags:
categories: ["深度学习笔记"]
mathjax: true
---
[You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) \
Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi \
University of Washington, Allen Institute for AI, Facebook AI Research \
http://pjreddie.com/yolo/

2016

**Abstract**
&emsp; We present YOLO, a new approach to object detection. 
Prior work on object detection repurposes classifiers to perform detection. 
先前关于目标检测的工作将重新定义分类器来执行检测。
Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. 
相反，我们将对象检测定义为一个回归问题，回归到空间分离的边界框和相关的类概率。
**A single neural network predicts bounding boxes and class probabilities directly from full images** in one evaluation(n.评价;[审计]评估). Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

&emsp; Our unified(unify v.整合,联合;统一) architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding(adj.令人震惊的;令人惊骇的) 155 frames per second while still `achieving double the mAP of other real-time detectors(达到其他实时检测器的两倍精度)`. Compared to state-of-the-art detection systems, **YOLO makes more localization errors but is less likely to predict false positives on background.** Finally, YOLO learns very general representations of objects. It outperforms(vt.胜过;做得比……好) other detection methods, including DPM and R-CNN, `when generalizing from natural images to other domains like artwork(当从自然图像推广到艺术作品等其他领域时)`.

# Introduction
&emsp; Humans glance at an image and instantly know what objects are in the image, where they are, and how they interact. The human visual system is fast and accurate, allowing us to perform complex tasks like driving with little conscious thought. 
Fast, accurate algorithms for object detection would allow computers to drive cars without specialized sensors, **enable** assistive(辅助的) devices **to** convey(vt.传达;运输) real-time scene information to human users, and unlock the potential for `general purpose(通用的)`, responsive(adj.响应的;应答的;响应灵敏的) robotic systems.
快速、准确的目标检测算法将允许计算机在没有专门传感器的情况下驾驶汽车，**使**辅助设备**能够向**人类用户传递实时的场景信息，并为通用、响应灵敏的机器人系统释放潜力。

&emsp; Current detection systems repurpose classifiers to perform detection. To detect an object, these systems take a classifier for that object and evaluate it at various locations and scales in a test image. Systems like deformable(可变形的) parts models (DPM) use a sliding window approach where the classifier is run at evenly(adv.均匀地;平衡地) spaced locations over the entire image [10].
**In short**: 当前分类器主要是基于滑动窗口，并在每个窗口做分类。

&emsp; More recent approaches like R-CNN use region proposal methods to first generate potential bounding boxes in an image and then run a classifier on these proposed boxes. After classification, `post-processing(后处理)` is used to refine(vt.精炼,提纯;改善) the bounding boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene [13]. These complex pipelines are **slow and hard to optimize** because each individual component must be trained separately.
**In short**: 最近的一些方法是像R-CNN这种使用RPN提取潜在的bbox，然后使用分类器对这些bbox做分类，最后再使用NMS做bbox改善。

&emsp; **We reframe**(v.给(照片)换框;再构造;全新地拟定) **object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities**. Using our system, you only look once (YOLO) at an image to predict what objects are present and where they are.

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/yolov1-1.jpg" width = 80% height = 80% />
</div>

&emsp; YOLO is refreshingly(adv.清爽地;有精神地) simple: see Figure 1. A single convolutional network simultaneously(同时地) predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance. This unified model has several benefits over traditional methods of object detection.

&emsp; First, YOLO is extremely fast. **Since we frame detection as a regression problem we don't need a complex pipeline**. We simply run our neural network on a new image at test time to predict detections. Our base network runs at 45 frames per second with no batch processing on a Titan X GPU and a fast version runs at more than 150 fps. This means we can process streaming video in real-time with less than 25 milliseconds of latency. Furthermore, YOLO achieves more than twice the mean average precision of other real-time systems. For a demo of our system running in real-time on a webcam please see our project webpage: http://pjreddie.com/yolo/.

&emsp; Second, **YOLO reasons globally about the image when making predictions**. Unlike sliding window and region proposal-based techniques, **YOLO sees the entire image** during training and test time **so it implicitly encodes contextual information about classes as well as their appearance**(外观;出现). **Fast R-CNN**, a top detection method [14], **mistakes background patches**(n.补丁;斑块(patch的复数);修补程序) **in an image for objects because it can't see the larger context**. YOLO makes less than half the number of background errors compared to Fast R-CNN.
**In short**: YOLO从全局去看待图像，这一点相比Fast R-CNN，YOLO把背景误检为目标几率较小。（也就是说Fast R-CNN更容易误检背景）
> 这似乎是说，相比Fast R-CNN，YOLO在查准率(Precision)方面较高。

&emsp; Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods like DPM and R-CNN `by a wide margin(大幅度地)`. Since YOLO is `highly generalizable(高度可泛化的)` it is less likely to `break down(发生故障,崩溃)` when applied to new domains or unexpected inputs.
**In short**: YOLO的泛化性能大幅领先于DPM和R-CNN系列。

&emsp; **YOLO still lags(v.落后于) behind state-of-the-art detection systems in accuracy**. While it can quickly identify objects in images **it struggles**(struggle v.奋斗;斗争;艰难地行进) to precisely localize some objects, **especially small ones**. We examine(审查;检查) these tradeoffs further in our experiments.
**In short**: YOLO在Accuracy方面仍然落后于SOTA检测系统，尽管YOLO很快但是它在小目标检测方面精度有待提高。
> 前面说相比Fast R-CNN，YOLO在查准率(Precision)方面较高，这里又说YOLO的Accuracy较SOTA低，看来YOLO的Recall也是较低的了。

&emsp; All of our training and testing code is open source. `A variety of(各种各样的)` pretrained models are also available to download.

# Unified Detection
&emsp; We unify(v.整合,联合;统一) the separate components of object detection into a single neural network. Our network **uses features from the entire image to predict each bounding box**. It also predicts all bounding boxes across all classes for an image simultaneously(adv.同时地). This means our network reasons globally about the full image and all the objects in the image. The YOLO design enables end-to-end training and realtime speeds while maintaining high average precision.

&emsp; **Our system divides the input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.**

&emsp; **Each grid cell predicts B bounding boxes and confidence scores for those boxes**. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. 
Formally(adv.正式地;形式上) **we define confidence as Pr(Object) \* $\text{IOU}^{truth}_{pred}$. If no object exists in that cell, the confidence scores should be zero**. 
形式上，我们定义置信度为 Pr(Object) \* $\text{IOU}^{truth}_{pred}$，如果cell中没有目标，则置信度为0. \
Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.
否则，我们希望置信度得分等于预测框与地面真实值之间的交并比(IOU)。

未完待续。。。