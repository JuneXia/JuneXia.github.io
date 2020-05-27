---
title: 
date: 2020-05-01
tags:
categories: ["深度学习笔记"]
mathjax: true
---

# 什么是anchor?
anchor最初是在Faster R-CNN中提出来，后来在YOLO、SSD中得到了广泛的应用。

feature-map 中的一个点在原图中的映射框就是anchor框(anchor-box)，而该anchor-box的中心点就是anchor，有些时候anchor和anchor-box是一个意思，具体可根据文章语境理解。（注意：anchor-box不是feature-map中的框，而是feature-map中的一个点对应原图中的映射框）
<!-- more -->


下面是我封装好的anchor框实现代码：
参考文献[2]
```python
import numpy as np


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    生成k个基础anchor框(在faster-rcnn中k等于9)
    :param base_size:
    :param ratios:
    :param anchor_scales:
    :return:
    """
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),
                      shift_x.ravel(),shift_y.ravel(),), axis=1)
    # shift 是一个 (N × 4) 的数组，而实际上(:, 0:2)和(:, 2:4)是相等的，这样做是为了下面计算的方便。

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        # 利用先验框和其对应的真实框进行编码
        loc = bbox2loc(anchor, bbox[argmax_ious])

        return loc, label

    def _create_label(self, anchor, bbox):
        # 1是正样本，0是负样本，-1忽略
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # argmax_ious为每个先验框对应的最大的真实框的序号
        # max_ious为每个真实框对应的最大的真实框的iou
        # gt_argmax_ious为每一个真实框对应的最大的先验框的序号
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox)

        # 如果小于门限函数则设置为负样本
        label[max_ious < self.neg_iou_thresh] = 0

        # 每个真实框至少对应一个先验框
        label[gt_argmax_ious] = 1

        # 如果大于门限函数则设置为正样本
        label[max_ious >= self.pos_iou_thresh] = 1

        # 判断正样本数量是否大于128，如果大于的话则去掉一些
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox):
        # 计算所有
        ious = bbox_iou(anchor, bbox)
        # 行是先验框，列是真实框
        argmax_ious = ious.argmax(axis=1)
        # 找出每一个先验框对应真实框最大的iou
        max_ious = ious[np.arange(len(anchor)), argmax_ious]
        # 行是先验框，列是真实框
        gt_argmax_ious = ious.argmax(axis=0)
        # 找到每一个真实框对应的先验框最大的iou
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 每一个真实框对应的最大的先验框的序号
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious
```

**代码示例1**: 下面代码演示一组anchors中的9个anchor框在输入图片中的尺度：
```python
import os
import cv2
import imageio
import numpy as np
from torchvision.models.mobilenet import mobilenet_v2
import torch
import anchors as Anchors
from libml.utils.config import SysConfig

if __name__ == '__main__':  # 演示一组anchors中的9个anchor框在输入图片中的尺度
    net = mobilenet_v2()

    base_size = 16  # anchor基础尺寸，当base_size=16时，anchor最小长或宽可低至 16*8*sqrt(0.5), 即16*8/sqrt(2)
                    # 如果希望对检测小尺寸目标有利，则应该要设置更小的 base_size，或者调低anchor_scales中的尺度。
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]

    # 生成一组anchor基，这组anchor基此时还没有中心点，它们只是对应着9种尺度。
    anchor_base = Anchors.generate_anchor_base(base_size=base_size, anchor_scales=anchor_scales, ratios=ratios)
    image_path = os.path.join(SysConfig["home_path"], SysConfig["proml_path"], "data/voc2007-train-000030.jpg")
    bboxes = np.array([[51, 225, 241, 474], [382, 335, 558, 472], [419, 261, 540, 477], [280, 220, 360, 300]])

    image = cv2.imread(image_path)
    image_size = (image.shape[0], image.shape[1])

    for box in bboxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('ground_truth', image)
    cv2.waitKey()

    # 作为示例，这里选择原图像的中心点作为anchor基的中心点，并画出一组anchor作为示例：
    anchor_center = np.array([image_size[0] / 2, image_size[1] / 2, image_size[0] / 2, image_size[1] / 2])

    # anchor_base + anchor_center 后就得到了这组anchor的坐标了
    anchors = (anchor_base + anchor_center).astype(np.int32)

    # 显示画框示例
    images = []
    for ach in anchors:
        cv2.rectangle(image, (ach[0], ach[1]), (ach[2], ach[3]), (180, 0, 0), 2)
        images.append(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
        cv2.imshow('anchor_example', image)
        cv2.waitKey()

    imageio.mimsave('./anchor_example.gif', images, 'GIF', duration=0.6)
    print('finish')
```

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/anchor_example.gif" width = 70% height = 70% />
</div>
<center>图2 &nbsp;  一组简单的anchor框示例</center>

上图展示了一组anchor框在原图中的尺度，而实际应用中这样的anchor组会依次铺满整副图像的。


**代码示例2：** 演示如何从整副图中的所有anchor框中提取与ground-truth有关的anchor框

```python
if __name__ == '__main__':  # 演示如何从整张图片中的所有anchor框中提取与ground-truth有关的anchor框
    net = mobilenet_v2()

    base_size = 16  # anchor基础尺寸，当base_size=16时，anchor最小长或宽可低至 16*8*sqrt(0.5), 即16*8/sqrt(2)
                    # 如果希望对检测小尺寸目标有利，则应该要设置更小的 base_size，或者调低anchor_scales中的尺度。
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]


    # step1: 生成anchor_base
    # ===========================================================
    # 生成一组anchor基，这组anchor基此时还没有中心点，它们只是对应着9种尺度。
    anchor_base = Anchors.generate_anchor_base(base_size=base_size, anchor_scales=anchor_scales, ratios=ratios)
    image_path = os.path.join(SysConfig["home_path"], SysConfig["proml_path"], "data/voc2007-train-000030.jpg")
    bboxes = np.array([[ 51, 225, 241, 474], [382, 335, 558, 472], [419, 261, 540, 477], [280, 220, 360, 300]])

    image = cv2.imread(image_path)
    image_size = (image.shape[0], image.shape[1])


    # step2: 计算backbone得到的特征图
    # ===========================================================
    # 由输入图片计算特征图
    t = np.expand_dims(image, axis=0).astype(np.float32)
    t = torch.tensor(t)
    t = t.permute((0, 3, 1, 2))
    features = net.features(t)

    # 提取特征图宽高
    feature_width = features.shape[2]
    feature_height = features.shape[3]

    # 计算在特征图上滑动anchor_base时的跨度
    # 一副图片经过backbone后得到的是多次下采样后的特征图，anchor框是指在输入图片上的anchor框.
    # 而要想在输入图片上均匀生成等间距的anchor_base，则需要有一个合理的anchor_base间隔，也就是下面即将要计算的feature_stride
    remainder = 1 if image_size[0] % feature_width > 0 else 0  # 有余数则为1，没有余数则为0
    feature_stride = image_size[0] // feature_width + remainder


    # step3: 根据特征图尺寸以及anchor_base生成正张输入图片上的anchors
    # ===========================================================
    # 生成所有的先验框：根据特征图宽、高和跨度对 anchor_base 进行平移，使其铺满至整副图片
    anchor = Anchors.enumerate_shifted_anchor(np.array(anchor_base), feature_stride, feature_height, feature_width)


    # step4: 通过nms从所有的anchor框提取和ground-truth有关的anchor框标签
    # ===========================================================
    # 上面得到的是输入图片上的所有anchor框，但实际上这些anchor框中大部分都不是我们想要的，
    # 我们想要的只是这些anchor框中和ground-truth有关的部分。
    # 通过nms从所有的anchor框提取和ground-truth有关的anchor框标签
    anchor_target_creator = Anchors.AnchorTargetCreator()
    argmax_ious, label = anchor_target_creator._create_label(anchor, bboxes)


    # 至此，所有的anchor框已经生成完毕，下面是可视化部分
    # ===========================================================
    # 画出ground-truth框
    for box in bboxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('ground-truth', image)
    cv2.waitKey()

    # 画出最终的anchor框(由于负样本太多，这里只画出了和正样本相关的anchor框)
    images = []
    anchor = anchor[np.where(label == 1)]
    for ach in anchor:
        cv2.rectangle(image, (ach[0], ach[1]), (ach[2], ach[3]), (0, 0, 200), 2)
        images.append(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
        cv2.imshow('pos-anchor', image)
        cv2.waitKey()

    imageio.mimsave('./pos-anchor.gif', images, 'GIF', duration=0.6)

    print('finish')
```

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/pos-anchor.gif" width = 70% height = 70% />
</div>
<center>图2 &nbsp;  与ground-truth有关的anchor框</center>


> 关于anchor框的一些值得注意的地方：
> 1. 无论ground-truth尺寸是大还是小，anchor_base中的9种尺寸都是固定的，这会导致像 图2 中间的小目标拥有较大的anchor框；\
> （这可以通过调节anchor_base的尺寸大小还适当匹配目标大小，也就是改变上述代码中的base_size或者anchor_scales）
> 2. 上述代码中计算feature_stride时的代码为：\
>    remainder = 1 if image_size[0] % feature_width > 0 else 0 \
>    feature_stride = image_size[0] // feature_width + remainder \
> 这里的feature_stride是anchor_base中心点在输入图片上滑动时每次移动的像素个数，而上述计算方式只是为了让生成的所有anchor框能够均匀分布在输入图片上。\
> （如果想要设计更加密集的anchor框，则可以适当减小feature_stride，但这应该是没有必要的，因为观察generate_anchor_base的计算代码可知，生成的anchor框的最小尺寸应该是$base_size \times feature_stride$，也就是说当base_size>1时，这个最小尺寸都比feature_stride大）



---------------------------------------------------------------
下面参考文献[1]，讲的并不是很好，记录一下，供参考吧。


# anchor-box 的生成

**定义输入及anchor框相关参数**
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

input_shape = (24, 24, 3)  # 输入图片尺寸
size_X = 3  # 每行生成多少个anchor框
size_Y = 3  # 每列生成多少个anchor框
rpn_stride = 8  # 相邻两个anchor框之间的跨度

scales = [1, 2, 4]  # anchor框的缩放比例
ratios = [0.5, 1, 2]  # anchor框的长宽比
```

**根据 缩放比(scales) 和 长宽比(ratios) 生成 meshgrid 坐标点**

3种缩放比和3种长宽比一共可以生成9种坐标点，其实就是对应这anchor框的9种尺寸。

```python
scales, ratios = np.meshgrid(scales, ratios)
"""
scales.shape = (3, 3)
scales:
array([[1, 2, 4],
       [1, 2, 4],
       [1, 2, 4]])

ratios.shape = (3, 3)
ratios: 
array([[0.5, 0.5, 0.5],
       [1. , 1. , 1. ],
       [2. , 2. , 2. ]])
"""

scales, ratios = scales.flatten(), ratios.flatten()
"""
scales.shape = (9,)
scales:
array([1, 2, 4, 1, 2, 4, 1, 2, 4])

scales.shape = (9,)
ratios:
array([0.5, 0.5, 0.5, 1. , 1. , 1. , 2. , 2. , 2. ])
"""
```
上面的scales可看成：
$
\text{scales} = (\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, 1, 1, 1, 2, 2, 2)
$


**根据定义的scales和ratios计算anchor框实际的长和宽**

```python
scalesY = scales * np.sqrt(ratios)
scalesX = scales / np.sqrt(ratios)
"""
scalesY.shape = (9,)
scalesY: 
array([0.70710678, 1.41421356, 2.82842712, 1.        , 2.        ,
       4.        , 1.41421356, 2.82842712, 5.65685425])

scalesX.shape = (9,)
scalesX: 
array([1.41421356, 2.82842712, 5.65685425, 1.        , 2.        ,
       4.        , 0.70710678, 1.41421356, 2.82842712])
"""
```
上面 scalesY 的计算过程：
$$
\text{scalesY} = (1 \cdot \frac{1}{\sqrt{2}}, 2 \cdot \frac{1}{\sqrt{2}}, 4 \cdot \frac{1}{\sqrt{2}}, 1, 2, 4, 1 \cdot \sqrt{2}, 2 \cdot \sqrt{2}, 4 \cdot \sqrt{2})
$$

上面 scalesX 的计算过程：
$$
\text{scalesX} = (1 \cdot \sqrt{2}, 2 \cdot \sqrt{2}, 4 \cdot \sqrt{2}, 1, 2, 4, 1 \cdot \frac{1}{\sqrt{2}}, 2 \cdot \frac{1}{\sqrt{2}}, 4 \cdot \frac{1}{\sqrt{2}})
$$
可以发现 scalesX 和 scalesY 在一定程度上对称的。


**计算每个anchor框的中心偏移量**

这里 anchor 框的中心偏移量是指 anchor 框的中心相对图像原点(0, 0)的偏移量。

```python
shiftX = np.arange(0, size_X) * rpn_stride  # 乘rpn_stride表示将anchor映射回原图
shiftY = np.arange(0, size_Y) * rpn_stride
"""
shiftX.shape = (3,)
shiftX:
array([ 0,  8, 16])

shiftY.shape = (3,)
shiftY:
array([ 0,  8, 16])
"""


shiftX, shiftY = np.meshgrid(shiftX, shiftY)
"""
shiftX.shape = (3, 3)
shiftX:
array([[ 0,  8, 16],
       [ 0,  8, 16],
       [ 0,  8, 16]])

shiftY.shape = (3, 3)
shiftY:
array([[ 0,  0,  0],
       [ 8,  8,  8],
       [16, 16, 16]])
"""
```


**将所有anchor中心偏移量和anchor长宽组合成meshgrid坐标**
```python
centerX, scalesX = np.meshgrid(shiftX, scalesX)
"""
centerX.shape = (9, 9)
centerX：
array([[ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16],
       [ 0,  8, 16,  0,  8, 16,  0,  8, 16]])

scalesX.shape = (9, 9)
scalesX：
array([[1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356,
        1.41421356, 1.41421356, 1.41421356, 1.41421356],
        ...
       [2.82842712, 2.82842712, 2.82842712, 2.82842712, 2.82842712,
        2.82842712, 2.82842712, 2.82842712, 2.82842712]])
"""

centerY, scalesY = np.meshgrid(shiftY, scalesY)
"""
centerY.shape = (9, 9)
centerY: 
array([[ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16],
       [ 0,  0,  0,  8,  8,  8, 16, 16, 16]])

scalesY.shape = (9, 9)
scalesY:
array([[0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678,
        0.70710678, 0.70710678, 0.70710678, 0.70710678],
       ...
       [5.65685425, 5.65685425, 5.65685425, 5.65685425, 5.65685425,
        5.65685425, 5.65685425, 5.65685425, 5.65685425]])
"""
```
如果 `np.meshgrid(X, Y)` 中的 X 或者 Y 是二维矩阵的话，则 np.meshgrid 内部会先将它们做 flatten 操作，然后再进行 meshgrid 操作。


**组合anchor中心坐标和anchor尺寸**
```python
anchor_center = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
"""
anchor_center.shape
Out[7]: (9, 9, 2)
"""

anchor_size = np.stack([anchorY, anchorX], axis=2).reshape(-1, 2)
"""
anchor_center.shape = (81, 2)
anchor_center:
array([[ 0,  0],
       [ 0,  8],
       [ 0, 16],
       [ 8,  0],
       [ 8,  8],
       [ 8, 16],
       [16,  0],
       [16,  8],
       [16, 16],
       ...
       [ 0,  0],
       [ 0,  8],
       [ 0, 16],
       [ 8,  0],
       [ 8,  8],
       [ 8, 16],
       [16,  0],
       [16,  8],
       [16, 16]])
"""
```


**根据anchor中心坐标和anchor尺寸计算anchor bounding-box**
```python
boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)
"""
anchor_size.shape = (9, 9, 2)
anchor_size:
array([[0.70710678, 1.41421356],
       [0.70710678, 1.41421356],
       [0.70710678, 1.41421356],
       ...
       [2.82842712, 5.65685425],
       [2.82842712, 5.65685425],
       [2.82842712, 5.65685425],
       ...
       [4.        , 4.        ],
       [4.        , 4.        ],
       [4.        , 4.        ],
       [1.41421356, 0.70710678],
       [1.41421356, 0.70710678],
       [1.41421356, 0.70710678],
       ...
       [5.65685425, 2.82842712],
       [5.65685425, 2.82842712],
       [5.65685425, 2.82842712]])
"""
```


**上述思想封装成完整的代码如下**：
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 常规输入以及anchor框参数
# *************************************
# input_shape = (128, 128, 3)
# size_X = 16  # 每行生成多少个anchor框
# size_Y = 16  # 每列生成多少个anchor框
# rpn_stride = 8  # 相邻两个anchor框之间的跨度
#
# scales = [2, 4, 8]  # anchor框的缩放比例
# ratios = [0.5, 1, 2]  # anchor框的长宽比
# *************************************


# 为了方便了解anchor框的生成机理，这里使用较小的参数来生成anchor框，方便理解
# *************************************
input_shape = (24, 24, 3)
size_X = 3  # 每行生成多少个anchor框
size_Y = 3  # 每列生成多少个anchor框
rpn_stride = 8  # 相邻两个anchor框之间的跨度

scales = [1, 2, 4]  # anchor框的缩放比例
ratios = [0.5, 1, 2]  # anchor框的长宽比
# *************************************


def anchor_gen(size_X, size_Y, rpn_stride, scales, ratios):
    scales, ratios = np.meshgrid(scales, ratios)
    scales, ratios = scales.flatten(), ratios.flatten()
    scalesY = scales * np.sqrt(ratios)
    scalesX = scales / np.sqrt(ratios)

    shiftX = np.arange(0, size_X) * rpn_stride
    shiftY = np.arange(0, size_Y) * rpn_stride
    shiftX, shiftY = np.meshgrid(shiftX, shiftY)

    centerX, anchorX = np.meshgrid(shiftX, scalesX)
    centerY, anchorY = np.meshgrid(shiftY, scalesY)

    anchor_center = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    anchor_size = np.stack([anchorY, anchorX], axis=2).reshape(-1, 2)

    boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)

    return boxes


if __name__ == '__main__':
    anchors = anchor_gen(size_X, size_Y, rpn_stride, scales, ratios)

    plt.figure(figsize=(10, 10))
    img = np.ones(input_shape)
    plt.imshow(img)

    axs = plt.gca()  # get current axs

    for i in range(anchors.shape[0]):
        box = anchors[i]
        rec = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor="r", facecolor="none")
        axs.add_patch(rec)

    plt.show()

    print('debug')
```

上面代码生成的anchor框：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/fasterrcnn_anchor1.jpg" width = 70% height = 70% />
</div>


由于上述代码中anchor框的中心是从(0, 0)开始计算的，所以有部分超出边界外了。

下面代码稍作修改：
```python
def anchor_gen(size_X, size_Y, rpn_stride, scales, ratios):
    ...

    shiftX = np.arange(0, size_X) * rpn_stride
    shiftY = np.arange(0, size_Y) * rpn_stride
    shiftX = shiftX + rpn_stride/2
    shiftY = shiftY + rpn_stride/2
    shiftX, shiftY = np.meshgrid(shiftX, shiftY)

    ...
```

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/fasterrcnn_anchor2.jpg" width = 70% height = 70% />
</div>



# 参考文献
[1] 网易云课堂 > 利用keras从头实现faster rcnn
[2] [bubbliiiing/faster-rcnn-pytorch](https://github.com/bubbliiiing/faster-rcnn-pytorch)