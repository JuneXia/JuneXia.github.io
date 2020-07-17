---
title: 
date: 2020-7-14
tags:
categories: ["深度学习笔记"]
mathjax: true
---

[Geometric Style Transfer](https://arxiv.org/abs/2007.05471)

Xiao-Chang Liu
University of Bath

Xuan-Yi Li  Ming-Ming Cheng
Nankai University
南开大学程明明团队

Peter Hall
University of Bath (巴斯大学)


**Abstract**
&emsp; Neural style transfer (NST), where an input image is rendered in the style of another image, has been a topic of considerable progress in recent years. Research over that time has been dominated by transferring aspects(aspect n.方面, 外貌, 外观; 方位, 方向) of color and texture, yet these factors are only one component of style. Other factors of style include composition(构图), the projection system used, and the way in which artists warp and bend(v.弯曲,屈服;n.弯曲(物),弯道) objects. **Our contribution is to introduce a neural architecture that supports transfer of geometric style.** Unlike recent work in this area, `we are unique in being general in that we are not restricted by semantic content (我们的独特之处在于它不受语义内容的限制)`. \

This new architecture runs prior to a network that transfers texture style, enabling us to transfer texture to a warped image. \
这种新架构在传递纹理样式的网络之前运行，从而使我们可以将纹理传递到扭曲的图像. \

This form of network supports a second novelty: we extend the NST input paradigm. Users can input content/style pair as is common, or they can chose to input a content/texture-style/geometry-style triple. \
第二点新颖性：我们扩展了NST输入范式。 用户可以输入常见的内容/样式对，也可以选择输入内容/纹理样式/几何样式三元组。\

This three image input paradigm divides style into two parts and so provides significantly greater versatility(n.多才多艺, 多样性，多功能) to the output we can produce. \
这三个图像输入范例 将 style 分为两个部分，因此 为 我们的输出提供了更大的通用性. \

We provide user studies that show the quality of our output, and quantify the importance of geometric style transfer to style recognition by humans. \
我们提供的用户研究可以显示输出的质量，并量化几何样式转移对人类样式识别的重要性。

