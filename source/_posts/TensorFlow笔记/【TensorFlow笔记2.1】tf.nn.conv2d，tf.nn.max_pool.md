---
title: 
date: 2017-10-26
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---

## tf.nn.conv2d函数解析
**tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)**

除去name参数用以指定该操作的name，与方法有关的一共五个参数：
<!-- more -->

第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一

第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维

第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4

第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）

第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true

结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。

那么TensorFlow的卷积具体是怎样实现的呢，用一些例子去解释它：

### 示例1：[1，3，3，1]图像，[1，1，1，1]卷积核
考虑一种最简单的情况，现在有一张3×3单通道的图像（对应的shape：[1，3，3，1]），用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，最后会得到一张3×3的feature map

```python
import tensorflow as tf

input = tf.Variable(tf.ones([1, 3, 3, 1]))
filter = tf.Variable(tf.ones([1, 1, 1, 1]))

with tf.device("/cpu:0"):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        conv2d_SAME = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)
        conv2d_VALID = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=False)
        ret_SAME = sess.run(conv2d_SAME)
        ret_VALID = sess.run(conv2d_VALID)
        print('ret_SAME.shape = ', ret_SAME.shape)
        print(ret_SAME)
        print('ret_VALID.shape = ', ret_VALID.shape)
        print(ret_VALID)
print('end')


输出结果：
ret_SAME.shape =  (1, 3, 3, 1)
[[[[1.]
   [1.]
   [1.]]
   
  [[1.]
   [1.]
   [1.]]
   
  [[1.]
   [1.]
   [1.]]]]
ret_VALID.shape =  (1, 3, 3, 1)
[[[[1.]
   [1.]
   [1.]]
   
  [[1.]
   [1.]
   [1.]]
   
  [[1.]
   [1.]
   [1.]]]]
```

### 示例2：[1，3，3，1]图像，[2，2，1，1]卷积核
输出图像不变，采用2x2的卷积核，其他不变
```python
filter = tf.Variable(tf.ones([2, 2, 1, 1]))

输出结果：
ret_SAME.shape =  (1, 3, 3, 1)
[[[[4.]
   [4.]
   [2.]]
   
  [[4.]
   [4.]
   [2.]]
   
  [[2.]
   [2.]
   [1.]]]]
ret_VALID.shape =  (1, 2, 2, 1)
[[[[4.]
   [4.]]
   
  [[4.]
   [4.]]]]
```

对于1x1的过滤器，无论padding是SAME还是VALID，其输出结果都很容易理解。下面主要说下对于2x2的过滤器。

对于3x3的图像，当过滤器是2x2，padding=SAME时，填充方式如下图(图片来自[1]，这里只看其填充方式，不要在意图中的具体数值)：
![enter image description here](https://lh3.googleusercontent.com/-bjuZbJNmYfI/W2MPCpHol4I/AAAAAAAAAEM/k45cej5ffe41ZbREtCSC2-Mr25vDcwIBQCLcBGAs/s0/conv2d_1.png "conv2d_1.png")

如果是2x2的图像，过滤器是3x3，padding=SAME，则填充方式如下图：
![enter image description here](https://lh3.googleusercontent.com/-j-WfJrhi8CY/W2MPs4D66hI/AAAAAAAAAEY/QDrVqWHgt_gQTC1Yqc1gG9VZxq-SeedmgCLcBGAs/s0/conv2d_2.png "conv2d_2.png")



### tf.nn.conv2d总结
下面以输入Tensor宽度为例(高度类似)简要说明在进行conv2d运算的时候，是如何在输入Tensor周围填充的。
（1）$input\_width=10，filter\_width=5，stride\_width=2$
&emsp; &emsp; 当padding='SAME'时，
![enter image description here](https://lh3.googleusercontent.com/-Kxiq_PH02fI/W2UylTE0_-I/AAAAAAAAAEs/Hm7cjyYSv7sKDjM376ovDpbGcPSLVO0twCLcBGAs/s0/conv2d_5.png "conv2d_5.png")

 - 此时$input\_width=10$是偶数，由于$stride\_width=2$，相当于是把$10$分成$10/2$份，也即：
$$卷积之后的宽度 = \frac{input\_width}{stride\_width}，当input\_width为偶数，padding=SAME时. \tag{1}$$
 - 对于第5份来说，需要补充$filter\_width-stride\_width=5-2$个0，也即：
$$input需要扩充的宽度=filter\_width-stride\_width，当input\_width为偶数，padding=SAME时. \tag{2}$$
注意实际操作时并不是将3个0都是放在后面的，而是拿出1个0到前面；

&emsp; &emsp; 当padding='VALID'时，此时不会对input周围进行填充，而只在input内部计算。而此时卷积后的宽度可以这样计算：另卷积后的宽度为n，当然n为正整数，且n必须满足不等式：
$$filter\_width+(n-1)*stride\_width≤input\_width，当padding=VALID时 \tag{3}$$
带入数值可解得$n≤3.5$，由于n为正整数，故n取3.
注：该不等式实际上是笔者根据n≤$\frac{input\_width-filter\_width}{stride\_width}+1$反推过来的。

（2）$input\_width=11，filter\_width=5，stride\_width=2$
&emsp; &emsp; 当padding='SAME'时，
![enter image description here](https://lh3.googleusercontent.com/-kMruiy2HpWs/W2UyqLUuuLI/AAAAAAAAAE0/xEInC1TXLX0_N8XXwcW9sA7A8cpF2noKwCLcBGAs/s0/conv2d_4.png "conv2d_4.png")

 - 此时$input\_width=11$是奇数，由于$stride\_width=2$，又由于padding='SAME'，所以要先把$input\_width$加1变成偶数，即在input后面添1个0，此时的$input\_width$相当于是12，这时候也就是把$12$分成$12/2$份，也即：
$$卷积之后的宽度 = \frac{input\_width+1}{stride\_width}，当input\_width为奇数，padding=SAME时. \tag{4}$$
 - 对于第6份来说，需要补充$filter\_width-stride\_width = 5-2$个0。
 - 注意：由于$input\_width$由11变成了12时添加了1个0，故这里一共添加了4个0，也即：
$$input需要扩充的宽度=filter\_width-stride\_width+1，当input\_width为奇数，padding=SAME时. \tag{5}$$
当然实际应用中会拿出4/2个0到前面去。

&emsp; &emsp; 当padding='VALID'时，计算方法同(3)式，此时解得卷积之后的宽度$n≤4$，n取正整数4.

（3）$input\_width=4，filter\_width=5，stride\_width=2$

 - 对于$filter\_widht>input\_width$这种情况来说，padding只能为'SAME'，padding='VALID'时会出错；
 - 计算方法同上。


## tf.nn.max\_pool函数解析

**tf.nn.max_pool(value, ksize, strides, padding, name=None)**   [2]
参数是四个，和卷积很类似：
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式

max\_pooling用法和conv2d类似，这里就不贴代码了。







## 参考文献
[1] [tf.nn.conv2d理解](https://blog.csdn.net/u013713117/article/details/55517458)
[2] 王晓华. TensorFlow深度学习应用实践
