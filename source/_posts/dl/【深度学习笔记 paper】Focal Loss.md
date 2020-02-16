---
title: 【深度学习笔记 paper】Focal Loss
date: 2019-04-05 17:28:05
tags:
categories: ["深度学习笔记"]
mathjax: true
---

论文：[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)


&emsp; Focal Loss 最初被提出是用来解决目标检测的[1]，在目标检测领域常见的算法主要可以分为两大类：two-stage detector和one-stage detector。前者是指类似Faster RCNN，RFCN这样需要region proposal的检测算法，这类算法可以达到很高的准确率，但是速度较慢。<!-- more -->
后者是指类似YOLO，SSD这样不需要region proposal，直接回归的检测算法，这类算法速度很快，但是准确率不如前者。作者提出focal loss的出发点也是希望one-stage detector可以达到two-stage detector的准确率，同时不影响原有的速度。

&emsp; 作者认为one-stage detector的准确率不如two-stage detector的原因是：样本的类别不均衡导致的，负样本数量太大，占总的loss的大部分，而且多是容易分类的，因此使得模型的优化方向并不是我们所希望的那样。因此针对类别不均衡问题，作者提出一种新的损失函数：focal loss，这个损失函数是在标准交叉熵损失基础上修改得到的。这个函数可以通过减少易分类样本的权重，使得模型在训练时更专注于难分类的样本。focal loss计算公式以及其和交叉熵的比较如下：

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/focalloss.jpg" width = 80% height = 80% />
</div>

其中CE是交叉熵，FL是focal loss，且当γ=0时，FL退化为CE。由上图可知，对于输出概率较大的易分类样本，focal loss计算得的数值几乎是0，这也就是说focal loss对易分类的样本不敏感，而更关注难分类的样本。

&emsp; Focal Loss 对于Object Detection来说可能效果不错，但我在人脸识别中引入Focal Loss时发现并没有多大作用。

&emsp; Focal Loss 算法实现，网上也有很多种[2]，但有的我实测存在数值不稳定问题，这里我做了一些改进并整理如下：

```python
# 交叉熵使用tf.nn.sparse_softmax_cross_entropy_with_logits实现
# 权重自己使用tf接口实现。感觉这个最好使！！！
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2., name='focal_loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # 从logits计算softmax
    reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
    prob = tf.nn.softmax(y_pred - reduce_max)

    # 计算交叉熵
    # clip_prob = tf.clip_by_value(prob, 1e-10, 1.0)
    # cross_entropy = -tf.reduce_sum(y_true * tf.log(clip_prob), 1)

    # 计算focal_loss
    prob = tf.reduce_max(prob, axis=1)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    loss = tf.reduce_mean(fl, name=name)

    return loss


# 损失波动较大
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2., name='focal_loss'):
    y_true = tf.one_hot(y_true, depth=y_pred.get_shape().as_list()[-1], dtype=tf.float32)

    reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
    y_pred = tf.nn.softmax(tf.subtract(y_pred, reduce_max))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
    # cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred), 1)
    cross_entropy = -tf.reduce_sum(tf.multiply(y_true, tf.log(y_pred)), axis=1)

    # 计算focal_loss
    prob = tf.reduce_max(y_pred, axis=1)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    # weight = tf.multiply(tf.multiply(weight, y_true), alpha)
    # weight = tf.reduce_max(weight, axis=1)

    fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    loss = tf.reduce_sum(fl, name=name)

    return loss


# 自己使用tf基础函数实现交叉熵
def focal_loss3(prediction_tensor, target_tensor, gamma=2., alpha=.25, name='focal_loss'):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    # y_true = tf.cast(target_tensor, tf.float32)
    y_true = tf.one_hot(target_tensor, depth=prediction_tensor.get_shape().as_list()[-1])
    y_pred = tf.clip_by_value(prediction_tensor, epsilon, 1. - epsilon)

    alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl, name=name)

    return loss

```



<!-- more -->

# 参考文献
[1] [Focal Loss](https://blog.csdn.net/u014380165/article/details/77019084)
[2] [focal loss的几种实现版本(Keras/Tensorflow)](https://blog.csdn.net/u011583927/article/details/90716942)
