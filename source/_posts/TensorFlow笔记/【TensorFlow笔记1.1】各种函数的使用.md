---
title: 
date: 2017-10-20
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---

<!-- more -->

### numpy随机数
参考资料[2]
np.random.randn(d0, d1, ..., dn)，从标准正太分布中产生shape为(d0, d1, ..., dn)的随机数组
np.random.rand(d0, d1, ..., dn)，从区间为[0, 1)的均匀分布中产生shape为(d0, d1, ..., dn)的随机数组


### 8.2 tensorflow常量、变量、数据类型、常用函数
tf.int8、tf.float32等tensorflow常用数据类型
tf.add、tf.mul等tensorflow常用函数

### 8.3 tensorflow矩阵计算
tf.random_normal、tf.truncated_normal、tf.random_uniform等随机生成矩阵张量（[1]中170页也有提及）

tf.diag、tf.diag_part、tf.trace、tf.transpose、tf.matmul、tf.matrix_determinant、tf.matrix_inverse、tf.cholesky、tf.matrix_solve

### 11.2.2 数据的矩阵化
tf.mul(matrix1, matrix2) #点乘，要求shape相同，对应相乘，现改用tf.multiply代替。
tf.matmul(matrix1, matrix2) # 叉乘，要求matrix1的列数等于matrix2的行数。类似numpy.dot(matrix1, matrix2)


### tf.argmax()
首先，明确一点，tf.argmax可以认为就是np.argmax。tensorflow使用numpy实现的这个API。
tf.argmax(input, axis=None, name=None, dimension=None, output_type=dtypes.int64):
**函数功能：**简单的说，就是返回最大值所在的下标[3]；
**参数解析：**
    axis：用于多维度计算
举例：
```
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]
```
axis=0 : 按列比较
```
test[0] = array([1, 2, 3])
test[1] = array([2, 3, 4])
test[2] = array([5, 4, 3])
test[3] = array([8, 7, 2])
# output   :    [3, 3, 1]      
```
axis=1 : 按行比较
```
test[0] = array([1, 2, 3])  #2
test[1] = array([2, 3, 4])  #2
test[2] = array([5, 4, 3])  #0
test[3] = array([8, 7, 2])  #0
```
这是里面都是数组长度一致的情况，如果不一致，axis最大值为最小的数组长度-1，超过则报错。
当不一致的时候，axis=0的比较也就变成了每个数组的和的比较。

### tf.reduce\_max和tf.reduce\_mean
求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
**参数解析：**
    input_tensor : 待求值的tensor；
    reduction_indices : 在哪一维上求解；
    keep_dims : 表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度。
例子：
```
# 'x' is [[1., 2.]
#         [3., 4.]]
```
以reduce_mean为例[4]：
```
tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
tf.reduce_mean(x, 0) ==> [2.,  3.] #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
tf.reduce_mean(x, 1) ==> [1.5,  3.5] #指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
tf.reduce_mean(x, 1, keep_dims=True) ==> [[1.5],  [3.5]]
```

> tf.reduce_mean中的keep\_dims有什么作用？
> 设有矩阵A和向量b如下：
> ```
> A = [[1., 2.],     b = [1., 1.]
>      [3., 4.]]
> ```
> 那么A-b是非法的，但若：
> ```
> b = [[1.], [1.]]，也即b = [[1.],
>                           [1.]]
> ```
> 那么此时A-b就是合法的了，此时：
> ```
> A-b=[[0., 1.],
>      [2., 3.]]
> ```

上述内容是以tf.reduce\_mean()为例，同理，还可用tf.reduce\_max()求最大值等。

### tf.equal
tf.equal(x, y, name=None)
例子：
```
import tensorflow as tf
import numpy as np
 
A = [[1,3,4,5,6]]
B = [[1,3,4,3,2]]
 
with tf.Session() as sess:
    print(sess.run(tf.equal(A, B)))
```
输出：
[[ True  True  True False False]]


### tf.cast
tf.cast(x, dtype, name=None)
**函数功能：**张量类型转换函数，将张量x的类型转换为dtype
例1：
```
# tensor `a` is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
```
例2：
```
a = tf.Variable([1,0,0,1,1])
b = tf.cast(a,dtype=tf.bool)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(b))
#[ True False False  True  True]
```

### tf.tile
tf.tile(input, multiples, name=None)
**函数功能：**
tile有平铺之意，用于在同一维度上的复制，用来对张量(Tensor)进行扩展，最终的输出张量的维度不变，但张量的shape会发生改变。
**参数解析：**
input, 输入的待扩展的张量；
multiples, 扩展方法，假如input是一个2维的张量。那么mutiples就必须是一个1x2的1维张量，这个张量的两个值依次表示input的第1、第2维数据扩展几倍。 
```
# 'x' is [[1., 2.]
#         [3., 4.]]
# tile = tf.tile(arr, [2])  # failed, multiples列表中的元素个数必须要和arr的维度相等
tile = tf.tile(arr, [2, 3])  ==>  [[1. 2. 1. 2. 1. 2.]
                                   [3. 4. 3. 4. 3. 4.]
                                   [1. 2. 1. 2. 1. 2.]
                                   [3. 4. 3. 4. 3. 4.]]
```



[1] 王晓华. TensorFlow深度学习应用实践 
[2] numpy随机数介绍https://blog.csdn.net/u013920434/article/details/52507173
[3] [tf.argmax()以及axis解析](https://blog.csdn.net/qq575379110/article/details/70538051/)
[4] [tf.reduce\_mean和tf.reduce\_max](https://blog.csdn.net/qq_32166627/article/details/52734387)
