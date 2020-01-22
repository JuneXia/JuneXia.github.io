---
title: 
date: 2017-10-23
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---

## 概述
&emsp; 除了典型的CSV文件存储方式外，TensorFlow还有专门的文件存储格式：TFRecords文件。
<!-- more -->

## TFRecords文件创建
&emsp; TFRecords文件一般用来存储特征值和其对应的标签。TFRecords文件中存储的内容是用通过 tf.train.Example 来创建的，我们可以将 tf.train.Example 创建的数据理解为sample(样本)。而 tf.train.Example 中的内容是通过 tf.train.Features 来创建的，tf.train.Features 中的内容是通过 tf.train.Feature 来创建的。

新建文件结构如下：
jpg
├── 001
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   ├── cat.3.jpg
│   ├── cat.4.jpg
│   ├── cat.5.jpg
│   ├── cat.6.jpg
│   ├── cat.7.jpg
│   ├── cat.8.jpg
│   └── cat.9.jpg
└── 002
    ├── dog.0.jpg
    ├── dog.1.jpg
    ├── dog.2.jpg
    ├── dog.3.jpg
    ├── dog.4.jpg
    ├── dog.5.jpg
    ├── dog.6.jpg
    ├── dog.7.jpg
    ├── dog.8.jpg
    └── dog.9.jpg

代码示例1（参考[1]例10-10）：
```python
import os
import tensorflow as tf
from PIL import Image

path = "jpg"
filenames=os.listdir(path)
writer = tf.python_io.TFRecordWriter("train.tfrecords")
# 同一个文件夹下的文件并不是按顺序来读取的，但一定会现将当前文件夹下的文件全部读完才会读下一个文件夹。

for name in os.listdir(path):
    class_path = path + os.sep + name
    for img_name in os.listdir(class_path):
        img_path = class_path+os.sep+img_name
        print(img_path)
        img = Image.open(img_path)
        img = img.resize((300,300))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
```


## TFRecords文件读取
&emsp; 要读取TFRecords的文件，使用 tf.TFRecordReader 与 tf.parse\_single\_example 解码器，然后使用 tf.FixedLengthRecordReader 和 tf.decode_raw 操作读取每个记录(即样本)[2]。

代码示例2（参考[1]例10-13）：
```python
import tensorflow as tf
import cv2
import numpy as np

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.decode_raw解码出来的Tensor还没有shape，tensorflow运算中需要的是有shape的张量。
    image = tf.reshape(image, [300, 300, 3])

    #image = tf.cast(image, tf.float32) * (1. / 128) - 0.5 # 归一化操作
    label = tf.cast(features['label'], tf.int32)
    # tf.cast(x, dtype, name=None), 类型转换函数，将x转换为dtype类型

    return image, label



filename = "train.tfrecords"
image, label = read_and_decode(filename)

# image_batch, label_batch = tf.train.batch([image, label], batch_size=1, num_threads=1, capacity=10) # 按顺序批处理
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=1, capacity=10, min_after_dequeue=3) # 随机批处理
# tf.train.shuffle_batch的capacity一定要比min_after_dequeue大

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

count = 0
for _ in range(100):
    count += 1
    # img, lab = sess.run([image, label]) # 只能正确输出9个样本，然后程序崩溃出错
    img, lab = sess.run([image_batch, label_batch]) # 能正确输出19个样本，然后程序崩溃出错
    img.resize((300, 300, 3))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) # val原本是用PIL.Image读取的，要想用opencv显示，则要将其转换为opencv的通道格式。
    #cv2.imshow("show", img)
    #cv2.waitKey()
    print(count, lab)

coord.request_stop()
coord.join(threads)
sess.close()
```

```python
语句1：img, lab = sess.run([image, label]) # 只能正确输出9个样本，然后程序崩溃出错
语句2：img, lab = sess.run([image_batch, label_batch]) # 能正确输出19个样本，然后程序崩溃出错
```

对于上述代码现象提出的疑问和自己猜测解释，如有错误还请指正。

疑问：相比语句2能正确输出19个样本，语句1为什么只能正确输出9个样本？
答：语句1直接sess.run的是filename\_queue中的结果，可能filename\_queue中的文件是并行输出的吧。而语句2中通过batch或shuffle\_batch将filename\_queue中的文件队列整合了一下。

疑问：无论是语句1还是语句2，为什么它们最后都崩溃出错了呢？
答：可能是没有设置成可以循环输入的方式吧。


## 参考文献
[1] 王晓华. TensorFlow深度学习应用实践 
[2] [ApacheCN >> Tensorflow >> 编程指南 >> 阅读数据](http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029497)
