---
title: 
date: 2018-09-16
tags:
categories: ["深度学习笔记"]
mathjax: true
---

论文：[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
Facebook AI Research (FAIR) & Cornell University and Cornell Tech
(Submitted on 9 Dec 2016 (v1), last revised 19 Apr 2017 (this version, v2))
<!-- more -->

**Abstract**
&emsp; Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive(密集,集中的,加强的). In this paper, we exploit the inherent(固有的;内在的;与生俱来的) multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal(微不足道的;边缘的,临界的) extra cost. A topdown architecture with lateral(侧面的,横向的) connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), `shows significant improvement as a generic feature extractor in several applications. (作为一种通用的特征提取器，它在一些应用中得到了显著的改进).` Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art singlemodel results on the COCO detection benchmark `without bells(bell:n.铃铛;v.系铃于…,鸣钟) and whistles(whistle:汽笛,哨子声)(没有任何附加条件)`, surpassing(surpass:优于,超出,卓越) all existing single-model entries(entry:进入,条目,记录) including those from the COCO 2016 challenge winners. In addition, our method can run at 6 FPS on a GPU and thus is a practical(实际的,实际用性的) and accurate solution to multi-scale object detection. Code will be made publicly available.

&emsp; Recognizing objects at vastly different scales is a fundamental challenge in computer vision. Feature pyramids `built upon(构建在…之上)` image pyramids (`for short(简称)` we call these featurized image pyramids) form the basis of a standard solution [1] (Fig. 1(a)). `These pyramids are scale-invariant in the sense that an object's scale change is offset(抵消,补偿,偏移量) by shifting(不断移动的,不断变化的. shift:v.转移,挪动;快速移动;变换) its level(水平,级别,电平,标准) in the pyramid. (这些金字塔是尺度不变的，因为一个物体的尺度变化被它在金字塔中的水平移动所抵消).` Intuitively(直观地,直觉地), this property enables a model to detect objects across a large range of scales by scanning the model over both positions and pyramid levels.

&emsp; Featurized image pyramids were heavily used in the era(时代,年代) of hand-engineered(手工设计的) features [5, 25]. `They were so critical(临界的,批评的,关键的) that object detectors like DPM [7] required dense scale sampling to achieve(实现,获得) good results (e.g., 10 scales per octave(八个一组的事物;八度)). (它们是如此的重要，以至于像DPM这样的目标探测器需要密集尺度的采样来获得良好的结果(例如，每八组10个尺度)).` For recognition tasks, engineered features have largely `been replaced with(被…取代/替换)` features computed by deep convolutional networks (ConvNets) [19, 20]. `Aside from(除…之外)` being capable of representing higher-level semantics, ConvNets are also more robust to variance in scale and thus facilitate(促进,帮助,使容易) recognition from features computed on a single input scale [15, 11, 29] (Fig. 1(b)). But even with this robustness, pyramids are still needed to get the most accurate results. All recent top entries in the ImageNet [33] and COCO [21] detection challenges `use multi-scale testing on featurized image pyramids (e.g., [16, 35]). (都在featurized image pyramids上使用多尺度测试).` `The principle(原理,原则;本质) advantage of featurizing each level of an image pyramid(对image pyramid的每一层都featurizing的本质优势)` is that it produces a multi-scale feature representation in which all levels are semantically strong, including the high-resolution levels.

&emsp; Nevertheless, featurizing each level of an image pyramid has obvious limitations. Inference time increases considerably (e.g., by four times [11]), making this approach impractical for real applications. Moreover, training deep networks end-to-end on an image pyramid is infeasible(不可行的) in terms of memory, `and so(因此,所以)`, if exploited, image pyramids are used only at test time [15, 11, 16, 35], `which creates an inconsistency between train/test-time inference. (这造成了训练/测试之间推断的不一致性).` For these reasons, Fast and Faster R-CNN [11, 29] opt to not use featurized image pyramids under default settings.



