---
title: 
date: 2020-04-11
tags:
categories: ["深度学习笔记"]
mathjax: true
---
Shape Robust Text Detection with Progressive Scale Expansion Network \
基于渐进式尺度扩展网络的形状鲁棒文本检测 \
Xiang Li, Wenhai Wang, Wenbo Hou, Ruo-Ze Liu, Tong Lu, Jian Yang \
DeepInsight@PCALab, 
Nanjing University of Science and Technology, 
National Key Lab for Novel Software Technology, 
Nanjing University \
CVPR2019
<!-- more -->


**研究背景** \
&emsp; 文章认为其提出的方法能**避免现有bounding box回归的方法产生的对弯曲文字的检测不准确的缺点**(如下图b所示) ,也能**避免现有的通过分割方法产生的对于文字紧靠的情况分割效果不好的缺点**(如下图c所示) 。该文章的网络框架是从FPN中受到启发采用了U形的网络框架,先通过将网络提取出的特征进行融合然后利用分割的方式将提取出的特征进行像素的分类,最后利用像素的分类结果通过一些后处理得到文本检测结果。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/PSENet1.jpg" width = 80% height = 80% />
</div>
<center>Figure 1: The results of different methods, best viewed in color. (a) is the original image. (b) refers to the result of bounding box regression-based method, which displays disappointing(disappoint v.使失望) detections as the red box covers nearly more than half of the context in the green box. (c) is the result of semantic segmentation, which mistakes the 3 text instances for 1 instance since their boundary pixels are partially connected. (d) is the result of our proposed PSENet, which successfully distinguishs and detects the 4 unique text instances.</center>

**研究成果** \
&emsp; 在ICDAR2015数据集上的最快能达到12.38fps。此时的f值为85.88%,而且该方法适用于弯曲文字的检测。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/PSENet2.jpg" width = 80% height = 80% />
</div>


**研究意义** \
&emsp; 通常OCR中,文字检测都是由目标检测继承而来,目标检测大多都是基于先验框的(anchor base),近期出现的no-anchor模式本质上也是基于先验框的。anchor-base模式在目标检测衍生到OCR领域就有很多缺陷,比如:倾斜(或扭曲)文字检测不准、过长文字串检测不全、过短文字串容易遗漏、距离较近的无法分开等缺点。渐进式扩展网络(PSENet)横空出世,以另一种思路解决了这些问题。该方法同样在工业应用中,很受欢迎,能够比较精准地解决实际问题。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/PSENet3.jpg" width = 80% height = 80% />
</div>

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/PSENet4.jpg" width = 80% height = 80% />
</div>

----------------------

**Abstract** \
&emsp; The challenges of shape robust text detection lie in two aspects: 1) most existing quadrangular bounding box based detectors are difficult to locate texts with arbitrary shapes, which are hard to be enclosed perfectly in a rectangle; 2) most pixel-wise segmentation-based detectors may not separate the text instances that are very close to each other. To address these problems, we propose a novel Progressive Scale Expansion Network (PSENet), designed as a segmentation-based detector with multiple predictions for each text instance. These predictions correspond to different kernels produced by shrinking the original text instance into various scales. Consequently, the final detection can be conducted through our progressive scale expansion algorithm which gradually expands the kernels with minimal scales to the text instances with maximal and complete shapes. Due to the fact that there are large geometrical margins among these minimal kernels, our method is effective to distinguish the adjacent text instances and is robust to arbitrary shapes. The state-of-the-art results on ICDAR 2015 and ICDAR 2017 MLT benchmarks further confirm the great effectiveness of PSENet. Notably, PSENet outperforms the previous best record by absolute 6.37% on the curve text dataset SCUT-CTW1500. Code will be available in https://github.com/whai362/PSENet.

**核心摘要**
1. 基于Bounding Box回归(Regression)的方法被提出了一组方法来成功地定位具有特定方向的矩形或四边形形式的文本目标。
2. 基于像素级别的语义分割的方法可以显式地处理曲线文本的检测问题。
3. 现有的基于回归的文本检测方法很难找到任意形状的文本,很难完全封闭在矩形中
4. 大多数基于像素的分割检测器可能不会将彼此非常接近的文本实例分开。
5. 针对任意形状的文本以及文本行无法区分的问题,本文提出了一种基于基于像素级别的分割的方法psenet,能够对任意形状的文本进行定位。提出一种渐进的尺度扩展算法,该算法可以成功地识别相邻文本实例

未完待续。。。

# 参考文献
[1] DeepShare.net \
[2] Shape Robust Text Detection with Progressive Scale Expansion Network
