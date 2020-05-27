---
title: 
date: 2020-05-06
tags:
categories: ["深度学习笔记"]
mathjax: true
---

- label 为 1 的 anchor: 当一个 anchor 与真实 bounding box 的最大 IOU 超过阈值 Vt1(0.7)
- label 为 -1 的 anchor :当一个 anchor 与真实 bounding box 的最大 I0U 低于阈值 Vt2(0.3)
- label 为 0 的 anchor : 当一个 anchor 与真实 bounding box 的最大 IOU 介于 Vt2 与 Vt1 之间
- Negative anchor 与 Positive anchor 的数量之和是一个人为设置的常数
- Input rpn bbox 输入的是 anchor 的回归量, RPN 网络计算的也是回归量
- 只有 positive anchor 才有对应的 Input rpn bbox

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/FasterRCNN-coding-rpn1.jpg" width = 80% height = 80% />
</div>


