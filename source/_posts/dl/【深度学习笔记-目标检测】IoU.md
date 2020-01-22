---
title: 【深度学习笔记-目标检测】IoU
date: 2018-04-05 17:28:05
tags:
categories: ["深度学习笔记"]
mathjax: true
---

## IoU
&emsp; Intersection over Union(简写：IoU)是一个评价指标，用于评价目标检测模型在特定数据集上的准确性，简单来讲就是模型产生的目标窗口和原来标记窗口(Ground Truth)的交叠率。一般的，IoU越大，表示准确性越高。
<!-- more -->

直观上来讲，准确度IoU计算公式如下：
<div align=center>![enter image description here](https://lh3.googleusercontent.com/-Zzs0Vg4o-70/XHikLJyHviI/AAAAAAAAAMo/5DnFXcCnFnUMVvxnarIDxN7xzVPwO1HcgCLcBGAs/s0/iou_equation.png=20x20)

<div align=center>![enter image description here](https://lh3.googleusercontent.com/--AZKkKgjPTU/XHinLjY6j_I/AAAAAAAAAM4/lIg7lpFtiQEX0FM1aQticq17qN5wKBCngCLcBGAs/s0/iou_stop_sign.jpg "iou_stop_sign.jpg")

图片来自文献[1]

### IoU计算方法
<center>![enter image description here](https://lh3.googleusercontent.com/-IUexEssoeYo/XHinpGdnSoI/AAAAAAAAANE/ibWcxZxIte4uZUXsTn3jwEU9C3U3W8UsACLcBGAs/s0/iou_%25E8%25AE%25A1%25E7%25AE%2597%25E6%2596%25B9%25E6%25B3%2595%25E5%259B%25BE%25E7%25A4%25BA.jpeg "iou_计算方法图示.jpeg")</center> 
(图片来自文献[2])

代码参考文献[1]
```python
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou
```

一般的，IoU越大，表示准确性越高，如下示例：
![enter image description here](https://lh3.googleusercontent.com/-aVMSFXQxiJU/XHipCZdOlzI/AAAAAAAAANY/fzGHMTiHpAogviwDOG3dv_ZcTWIrM7NHQCLcBGAs/s0/iou_examples.png "iou_examples.png")
（图片来自文献[1]）



## 参考文献
[1] [Intersection over Union (IoU) for object detection](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
[2] [目标识别（object detection）中的 IoU（Intersection over Union）](https://blog.csdn.net/lanchunhui/article/details/71190055)
