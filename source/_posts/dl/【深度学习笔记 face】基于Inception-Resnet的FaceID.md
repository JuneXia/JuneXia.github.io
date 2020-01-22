---
title: 【深度学习笔记 face】基于Inception-Resnet的FaceID
date: 2018-06-25
tags:
categories: ["深度学习笔记"]
mathjax: true
---


# Inception-Resnet-V2网络结构
> 因markdown 语法绘制表格不能实现单元格合并，故这里将部分Inception和Residual结构统一放到一个表格中，仅为个人记录学习所用。
<!-- more -->


本篇网络结构在原论文的基础上稍有改动。

| repeat | name | branch | shape | num outputs | kernel size | stride | padding | describe |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| | inputs | | (-1,160,160,3) | | | | | |
| | Conv2d_1a_3x3 | | (-1,79,79,32) | 32 | 3 | 2 | VALID | 损失1行1列像素 |
| | Conv2d_2a_3x3 | | (-1,77,77,32) | 32 | 3 | 1 | VALID |
| | Conv2d_2b_3x3 | | (-1,77,77,64) | 64 | 3 | 1 | SAME | 补充2行2列像素 |
| | MaxPool_3a_3x3 | | (-1,38,38,64) | - | 3 | 2 | VALID |
| | Conv2d_3b_1x1 | | (-1,38,38,80) | 80 | 1 | 1 | VALID | 上采样 |
| | Conv2d_4a_3x3 | | (-1,36,36,192) | 192 | 3 | 1 | VALID | |
| | MaxPool_5a_3x3 | | (-1,17,17,192) | - | 3 | 2 | VALID | 损失1行1列像素 |
| | Mixed_5b | Branch_0 Conv2d_1x1 | (-1,17,17,96) | 96 | 1 | 1 | SAME |
| | Mixed_5b | Branch_1 Conv2d_0a_1x1 | (-1,17,17,48) | 48 | 1 | 1 | SAME |
| | Mixed_5b | Branch_1 Conv2d_0b_5x5 | (-1,17,17,64) | 64 | 5 | 1 | SAME | 补充4行4列像素 |
| | Mixed_5b | Branch_2 Conv2d_0a_1x1 | (-1,17,17,64) | 64 | 1 | 1 | SAME |
| | Mixed_5b | Branch_2 Conv2d_0b_3x3 | (-1,17,17,96) | 96 | 3 | 1 | SAME | 补充2行2列像素 |
| | Mixed_5b | Branch_2 Conv2d_0c_3x3 | (-1,17,17,96) | 96 | 3 | 1 | SAME | 补充2行2列像素 |
| | Mixed_5b | Branch_3 AvgPool_0a_3x3 | (-1,17,17,192) | - | 3 | 1 | SAME | 补充2行2列像素 |
| | Mixed_5b | Branch_3 Conv2d_0b_1x1 | (-1,17,17,64) | 64 | 1 | 1 | SAME |
| | Mixed_5b | concat | (-1,17,17,320) | - | - | - | - |
| 10 | &darr;&emsp; Block35 | Branch_0 Conv2d_1x1 | (-1,17,17,32) | 32 | 1 | 1 | SAME |
| 10 | &darr;&emsp; Block35 | Branch_1 Conv2d_0a_1x1 | (-1,17,17,32) | 32 | 1 | 1 | SAME |
| 10 | &darr;&emsp; Block35 | Branch_1 Conv2d_0b_3x3 | (-1,17,17,32) | 32 | 3 | 1 | SAME | 补充2行2列像素 |
| 10 | &darr;&emsp; Block35 | Branch_2 Conv2d_0a_1x1 | (-1,17,17,32) | 32 | 1 | 1 | SAME |
| 10 | &darr;&emsp; Block35 | Branch_2 Conv2d_0b_3x3 | (-1,17,17,48) | 48 | 3 | 1 | SAME | 补充2行2列像素 |
| 10 | &darr;&emsp; Block35 | Branch_2 Conv2d_0c_3x3 | (-1,17,17,64) | 64 | 3 | 1 | SAME | 补充2行2列像素 |
| 10 | &darr;&emsp; Block35 | concat | (-1,17,17,128) | - | - | - | - |
| 10 | &darr;&emsp; Block35 | Conv2d_1x1 | (-1,17,17,320) | 320 | 1 | 1 | SAME |
| 10 | Block35 | input + scale*up | (-1,17,17,320) | - | - | - | - |
| | Mixed_6a | Branch_0 Conv2d_1a_3x3 | (-1,8,8,384) | 384 | 3 | 2 | VALID | 此处feature map17x17是奇数，不损失像素 |
| | Mixed_6a | Branch_1 Conv2d_0a_1x1 | (-1,17,17,256) | 256 | 1 | 1 | SAME |
| | Mixed_6a | Branch_1 Conv2d_0b_3x3 | (-1,17,17,256) | 256 | 3 | 1 | SAME | 补充2行2列像素 |
| | Mixed_6a | Branch_1 Conv2d_1a_3x3 | (-1,8,8,384) | 384 | 3 | 2 | VALID | 此处feature map17x17是奇数，不损失像素 |
| | Mixed_6a | Branch_2 MaxPool_1a_3x3 | (-1,8,8,320) | - | 3 | 2 | VALID | 此处feature map17x17是奇数，不损失像素 |
| | Mixed_6a | Branch_2 concat | (-1,8,8,1088) | - | - | - | - |
| 20 | &darr;&emsp; Block17 | Branch_0 Conv2d_1x1 | (-1,8,8,192) | 192 | 1 | 1 | SAME |
| 20 | &darr;&emsp; Block17 | Branch_1 Conv2d_0a_1x1 | (-1,8,8,128) | 128 | 1 | 1 | SAME |
| 20 | &darr;&emsp; Block17 | Branch_1 Conv2d_0b_1x7 | (-1,8,8,160) | 160 | [1,7] | 1 | SAME | 补充6列像素 |
| 20 | &darr;&emsp; Block17 | Branch_1 Conv2d_0c_7x1 | (-1,8,8,192) | 192 | [7,1] | 1 | SAME | 补充6行像素 |
| 20 | &darr;&emsp; Block17 | concat | (-1,8,8,384) | - | - | - | - |
| 20 | &darr;&emsp; Block17 | Conv2d_1x1 | (-1,8,8,1088) | 1088 | 1 | 1 | SAME |
| 20 | Block17 | input + scale*up | (-1,8,8,1088) | - | - | - | - |
| | Mixed_7a | Branch_0 Conv2d_0a_1x1 | (-1,8,8,256) | 256 | 1 | 1 | SAME |
| | Mixed_7a | Branch_0 Conv2d_1a_3x3 | (-1,3,3,384) | 384 | 3 | 2 | VALID | 损失1行1列像素 |
| | Mixed_7a | Branch_1 Conv2d_0a_1x1 | (-1,8,8,256) | 256 | 1 | 1 | SAME |
| | Mixed_7a | Branch_1 Conv2d_1a_3x3 | (-1,3,3,288) | 288 | 3 | 2 | VALID | 损失1行1列像素 |
| | Mixed_7a | Branch_2 Conv2d_0a_1x1 | (-1,8,8,256) | 256 | 1 | 1 | SAME |
| | Mixed_7a | Branch_2 Conv2d_0a_1x1 | (-1,8,8,288) | 288 | 3 | 1 | SAME | 补充2行2列像素 |
| | Mixed_7a | Branch_2 Conv2d_0a_1x1 | (-1,3,3,320) | 320 | 3 | 2 | VALID | 损失1行1列像素 |
| | Mixed_7a | Branch_3 MaxPool_1a_3x3 | (-1,3,3,1088) | - | 3 | 2 | VALID | 损失1行1列像素 |
| | Mixed_7a | concat | (-1,3,3,2080) | - | - | - | - |
| 10 | &darr;&emsp; Block8 | Branch_0 Conv2d_1x1 | (-1,3,3,192) | 192 | 1 | 1 | SAME |
| 10 | &darr;&emsp; Block8 | Branch_1 Conv2d_0a_1x1 | (-1,3,3,192) | 192 | 1 | 1 | SAME |
| 10 | &darr;&emsp; Block8 | Branch_1 Conv2d_0b_1x3 | (-1,3,3,224) | 224 | [1,3] | 1 | SAME | 补充2列像素 |
| 10 | &darr;&emsp; Block8 | Branch_1 Conv2d_0c_3x1 | (-1,3,3,256) | 256 | [3,1] | 1 | SAME | 补充2行像素 |
| 10 | &darr;&emsp; Block8 | concat | (-1,3,3,448) | - | - | - | - |
| 10 | &darr;&emsp; Block8 | Conv2d_1x1 | (-1,3,3,2080) | 2080 | 1 | 1 | SAME |
| 10 no activate at last | Block8 | input + scale*up | (-1,3,3,2080) | - | - | - | - |
| | Conv2d_7b_1x1 | | (-1,3,3,1536) | 1536 | 1 | 1 | SAME |
| | Logits AvgPool_1a_8x8 | | (-1,1,1,1536) | - | 1 | 1 | VALID |
| | Logits flatten | | (-1,1536) | - | - | - | - |
| | Logits Dropout | | (-1,1536) | - | - | - | - |
| | Bottleneck fully_connected | | (-1,1024) | - | - | - | - |


# 参考文献
[1] [davidsandberg/facenet](https://github.com/davidsandberg/facenet)





