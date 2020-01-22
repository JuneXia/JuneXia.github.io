---
title: 
date: 2017-10-03
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---
<!-- more -->

# 张量限幅
## clip_by_value
tf.maximum
tf.minimum
tf.clip_by_value(tensor, 2, 8)

## relu
clip_by_norm
gradient clipping

## Gradient clipping
梯度裁减，解决Gradient Exploding or vanishing. 
参考[1].47.张量限幅2

## clip_by_norm

### tf.clip_by_norm
改变一个Tensor的大小，但又不改变其方向
```python
a = tf.random.normal([2, 2], mean=10)
# tf.norm(a)
aa = tf.clip_by_norm(a, n)
```

### tf.clip_by_global_norm
对所有参数进行缩放



# 参考文献
[1] 龙龙深度学习


