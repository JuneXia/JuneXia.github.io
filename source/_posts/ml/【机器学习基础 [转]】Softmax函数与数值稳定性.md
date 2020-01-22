---
title: 【机器学习基础 [转]】Softmax函数与数值稳定性
date: 2018-06-19
tags:
categories: ["机器学习笔记"]
mathjax: true
---

## Softmax函数
本文参考文献[1].

&emsp; 在Logistic regression二分类问题中，我们可以使用sigmoid函数将输入$\boldsymbol Wx + \boldsymbol b$映射到(0, 1)区间中，从而得到属于某个类别的概率。将这个问题进行泛化，推广到多分类问题中，我们可以使用softmax函数，对输出的值归一化为概率值。
<!-- more -->

这里假设在进入softmax函数之前，已经有模型输出C值，其中C是要预测的类别数，模型可以是全连接网络的输出a，其输出个数为C，即输出为$a_{1}, a_{2}, ..., a_{C}$。

所以对每个样本，它属于类别i的概率为：

$$y_{i} = \frac{e^{a_i}}{\sum_{k=1}^{C}e^{a_k}} \ \ \ \forall i \in 1...C$$

通过上式可以保证$\sum_{i=1}^{C}y_i = 1$，即属于各个类别的概率和为1。



## softmax的计算与数值稳定性

用numpy实现的softmax函数为：
```python
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
```

传入[1, 2, 3, 4, 5]的向量
```python
>>> softmax([1, 2, 3, 4, 5])
array([ 0.01165623,  0.03168492,  0.08612854,  0.23412166,  0.63640865])
```

但如果输入值较大时：
```python
>>> softmax([1000, 2000, 3000, 4000, 5000])
array([ nan,  nan,  nan,  nan,  nan])
```
这是因为在求exp(x)时候溢出了：

&emsp; 一种简单有效避免该问题的方法就是让exp(x)中的x值不要那么大或那么小，在softmax函数的分式上下分别乘以一个非零常数：

$$y_{i} = \frac{e^{a_i}}{\sum_{k=1}^{C}e^{a_k}}= \frac{Ee^{a_i}}{\sum_{k=1}^{C}Ee^{a_k}}= \frac{e^{a_i+log(E)}}{\sum_{k=1}^{C}e^{a_k+log(E)}}= \frac{e^{a_i+F}}{\sum_{k=1}^{C}e^{a_k+F}}$$

这里$log(E)$是个常数，所以可以令它等于$F$。加上常数$F$之后，等式与原来还是相等的，所以我们可以考虑怎么选取常数$F$。我们的想法是让所有的输入在0附近，这样$e^{a_i}$的值不会太大，所以可以让F的值为：
$$F = -max(a_1, a_2, ..., a_C)$$
这样子将所有的输入平移到0附近（当然需要假设所有输入之间的数值上较为接近），同时，除了最大值，其他输入值都被平移成负数，e为底的指数函数，越小越接近0，这种方式比得到nan的结果更好。
```python
def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)

>>> softmax([1000, 2000, 3000, 4000, 5000])
array([ 0.,  0.,  0.,  0.,  1.])
```

当然这种做法也不是最完美的，因为softmax函数不可能产生0值，但这总比出现nan的结果好，并且真实的结果也是非常接近0的。


由于numpy是会溢出的，所以使用Python中的bigfloat库。
```python
import bigfloat

def softmax_bf(x):
	exp_x = [bigfloat.exp(y) for y in x]
	sum_x = sum(exp_x)
	return [y / sum_x for y in exp_x]

res = softmax_bf([1000, 2000, 3000, 4000, 5000])
print('[%s]' % ', '.join([str(x) for x in res]))
```
结果：
[6.6385371046556741e-1738, 1.3078390189212505e-1303, 2.5765358729611501e-869, 5.0759588975494576e-435, 1.0000000000000000]

可以看出，虽然前四项结果的量级不一样，但都是无限接近于0，所以加了一个常数的softmax对原来的结果影响很小。



在tensorflow中我们可以直接调用tf.nn.softmax来实现softmax功能，这里对tf.nn.softmax做了些数值稳定方面的改进：
```python
def softmax_NumStability(x):
    reduce_max = tf.reduce_max(x, 1, keepdims=True)
    prob = tf.nn.softmax(x - reduce_max)
    return prob
```


## 参考文献
[1] [Softmax函数与交叉熵](https://zhuanlan.zhihu.com/p/27223959)

