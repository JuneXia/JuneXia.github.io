---
title: 
date: 2017-10-22
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---

<!-- more -->

## tensorflow队列
&emsp; 在tensorflow中可以使用FIFOQueue、RandomShuffleQueue等方式创建一个队列[1]。

代码示例1：
```python
import tensorflow as tf

with tf.Session() as sess:
    q = tf.FIFOQueue(3, "float") # 创建长度为3，元素数据类型是float的队列。
    init = q.enqueue_many(([0.1, 0.2, 0.3],)) # 向队列中填充数据（注意这只是预备操作，真正的数据填充是要到sess.run(init)操作时才会完成）
    init2 = q.dequeue() # 出队
    init3 = q.enqueue(1.) # 入队

    sess.run(init)
    sess.run(init2)
    sess.run(init3)

    quelen =  sess.run(q.size())
    for i in range(quelen):
        print(sess.run(q.dequeue()))
```

## tensorflow中队列如何实现入队与出队同时进行
&emsp; 上述代码是现将所有数据都存入队列，然后再依次从队列中取出，这并没有发挥出队列的价值。队列是为了实现入队与出队操作可以同时进行而设计的，tensorflow中可以通过QueueRunner和Coordinator协作来实现这项工作。
下面先简要说下tf.train.Coordinator和tf.train.QueueRunner的用法和意义。
```python
# 创建线程协调器，用于协调主线程和各个子线程之间的交互操作。
coord = tf.train.Coordinator()
    
queue_runner = tf.train.QueueRunner(q, enqueue_ops=[add_op, enqueue_op] * 2) # 定义用2个线程去完成这项任务
# 先用QueueRunner定义队列的入队操作，然后用queue_runner创建子线程去处理该入队操作。
# queue_runner在创建线程的时候需要传入Coordinator协调器，用于和主线程协调操作。
enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)  # 启动入队线程
# queue_runner在创建线程的时候如果不传入Coordinator协调器的话，则程序运行结束前会报错。
# 这是因为当主线程运行完毕后就接直接结束了，而没有发出终止其他线程的请求。
```


完整代码如下，代码示例2：
```python
import tensorflow as tf

with tf.Session() as sess:
    q = tf.FIFOQueue(10, "float32") # 创建一个队列，该队列有10个数据，数据类型是float32
    counter = tf.Variable(0.0)
    add_op = tf.assign_add(counter, tf.constant(1.0))
    enqueue_op = q.enqueue(counter)

    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    queue_runner = tf.train.QueueRunner(q, enqueue_ops=[add_op, enqueue_op] * 2)
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)  # 启动入队线程

    for i in range(100):
        print(sess.run(q.dequeue()))
    coord.request_stop()
    coord.join(enqueue_threads)
    print('sess end')
print('program end')
```

## CSV文件读写
&emsp; 逗号分隔值（Comma-Separated Values，CSV，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）[1]。
在介绍tensorflow读写csv文件之前，先说下python读写csv文件
### Python读写CSV文件
#### Python写CSV文件
新建img文件夹，并放入若干张图片，如下图所示：
img
├── cat.0.jpg
├── cat.1.jpg
├── cat.2.jpg
├── cat.3.jpg
├── cat.4.jpg
├── cat.5.jpg
├── cat.6.jpg
├── cat.7.jpg
├── cat.8.jpg
└── cat.9.jpg
下面介绍python写csv文件。

代码示例:3：
```python
import os
path = 'img'
filenames=os.listdir(path)
strText = ""

with open("train_list.csv", "w") as fid:
    for a in range(len(filenames)):
        strText = path+os.sep+filenames[a]  + "," + filenames[a].split('.')[1]  + "\n"
        fid.write(strText)
fid.close()
```
生成的csv文件内容如下：
img/cat.0.jpg,0
img/cat.8.jpg,8
img/cat.3.jpg,3
img/cat.2.jpg,2
img/cat.1.jpg,1
img/cat.5.jpg,5
img/cat.4.jpg,4
img/cat.7.jpg,7
img/cat.6.jpg,6
img/cat.9.jpg,9

#### Python读取CSV文件

代码示例4：
```python
import tensorflow as tf
import cv2

image_add_list = []
image_label_list = []
with open("train_list.csv") as fid:
    for image in fid.readlines():
        image_add_list.append(image.strip().split(",")[0])
        image_label_list.append(image.strip().split(",")[1])

# 上面这段代码就是csv文件的读取，
# 下面介绍一下如何将图片文件转换成tensorflow所需要的张量形式。

def get_image(image_path):
    return tf.image.convert_image_dtype(
        tf.image.decode_jpeg(
            tf.read_file(image_path), channels=1),
        dtype=tf.uint8)
# tf.read_file, 读取图片文件
# tf.image.decode_jpeg, 将读取进来的图片文件解码成jpg格式
#                       channels=1表示读取灰度图
# tf.image.convert_image_dtype，将图像转化成TensorFlow需要的张量形式

img = get_image(image_add_list[0])

with tf.Session() as sess:
    cv2Img = sess.run(img)
    img2 = cv2.resize(cv2Img, (200,200))
    cv2.imshow('image', img2)
    cv2.waitKey()
```


### tensorflow读写CSV文件
&emsp; 关于CSV文件的读写，文献[1]中介绍的是用Python写CSV，用Python读CSV；文献[2]中介绍的是用tensorflow读取CSV。所以，有用tensorflow写CSV吗？好吧，遇到时再说吧。

&emsp; 新建文件file0.csv、file1.csv，其内容分别如下：
file0.csv
21,31,41,44,0
22,32,42,44,0
23,33,53,44,0
24,34,44,44,0
25,35,45,44,0

file1.csv
11,31,41,50,1
12,42,42,55,1
13,23,53,55,1
14,34,44,45,1
15,35,45,55,1


#### tensorflow读取CSV文件
使用 tf.TextLineReader与 tf.decode_csv操作，主要代码讲解如下：
```python
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
# 将文件名列表传递给tf.train.string_input_producer函数。string_input_producer创建一个用于保存文件名的FIFO队列。
```
```
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
# decode_csv操作将value解析成张量列表，record_defaults参数决定了所得张量的类型。
# 注意，如果要读取的每个记录是固定数量字节的二进制文件（这个一般是TFRecords文件而不是csv文件了吧），请使用 tf.FixedLengthRecordReader 读取该文件，并使用 tf.decode_raw 解码文件内容。decode_raw 操作进行从字符串到UINT8张量转换。
```

```python
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator() # 创建线程协调器
    threads = tf.train.start_queue_runners(coord=coord) # 启动线程，用于往队列中输入数据
    # 对比“代码示例2”中的tf.train.QueueRunner和queue_runner.create_threads，这里用tf.train.start_queue_runners包含了这两步操作
    # 疑问：为什么这里的tf.train.start_queue_runners没有传入sess参数？
    # 这可能是因为tf.train.start_queue_runners是被包含在“with tf.Session() as sess”里的吧
```

完整代码如下[2]，代码示例5：
```python
import tensorflow as tf

filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue) # read操作每次从文件中读取一行
# key是文件名，value是该文件中某一行内容，这些可以在后面通过sess.run查看

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator() # 创建线程协调器
    threads = tf.train.start_queue_runners(coord=coord) # 启动线程，用于往队列中输入数据
    # 注意：如果不启动该线程，则不会有往队列输入数据的操作，则下面的sess.run(...)会一直被阻塞

    for i in range(1200):
        # Retrieve a single instance:
        example, label = sess.run([features, col5])
        print(example, label)

    coord.request_stop()
    coord.join(threads)
```

## 参考文献
[1] 王晓华. TensorFlow深度学习应用实践 
[2] [ApacheCN >> Tensorflow >> 编程指南 >> 阅读数据](http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029497)
