---
title: 
date: 2017-10-15
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---
<!-- more -->

# tf.image下的各种resize
```python
import tensorflow as tf  
import cv2  
import numpy as np  
  
  
if __name__ == '__main__':  
    image_size = (300, 400)  
    filename = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44/n000020/0001_01.png'  
  file_contents = tf.read_file(filename)  
    # image = tf.image.decode_image(file_contents, 3)  
  image = tf.image.decode_png(file_contents, 3)  
    rawimage = tf.identity(image)  
  
    resize_images = tf.image.resize_images(image, image_size, align_corners=True, preserve_aspect_ratio=True)  # 如果preserve_aspect_ratio为True，则保持宽高比对原图进行缩放，缩放后的图像宽或高等于image_size中的最小值  
  resize_images = tf.cast(resize_images, dtype=tf.uint8)  
  
    # 双三次插值  
  resize_bicubic = tf.image.resize_bicubic(tf.expand_dims(image, axis=0), image_size)  # 实测如果设置 half_pixel_centers 参数，会报错  
  resize_bicubic = tf.cast(resize_bicubic, dtype=tf.uint8)  
  
    # 双线性插值  
  resize_bilinear = tf.image.resize_bilinear(tf.expand_dims(image, axis=0), image_size)  # 实测如果设置 half_pixel_centers 参数，会报错  
  resize_bilinear = tf.cast(resize_bilinear, dtype=tf.uint8)  
  
    # 最近邻插值  
  resize_nearest_neighbor = tf.image.resize_nearest_neighbor(tf.expand_dims(image, axis=0), image_size)  # 实测如果设置 half_pixel_centers 参数，会报错  
  
  # 区域插值  
  resize_area = tf.image.resize_area(tf.expand_dims(image, axis=0), image_size)  # 基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹,但是当图像放大时,它的效果与INTER_NEAREST效果相似.  
  resize_area = tf.cast(resize_area, dtype=tf.uint8)  
  
    init = tf.initialize_all_variables()  
  
    with tf.Session() as sess:  
        sess.run(init)  
  
        cv2.imshow('rawimg', sess.run(rawimage))  
  
        cv2.imshow('resize_images', sess.run(resize_images))  
        cv2.imshow('resize_bicubic', sess.run(resize_bicubic)[0])  
        cv2.imshow('resize_bilinear', sess.run(resize_bilinear)[0])  
        cv2.imshow('resize_nearest_neighbor', sess.run(resize_nearest_neighbor)[0])  
        cv2.imshow('resize_area', sess.run(resize_area)[0])  
  
        cv2.waitKey(0)
```


# tf.image下的各种resize、crop、pad
```python
if __name__ == '__main__':  # resize_crop_pad  
  image_size = (128, 128)  
    filename = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44/n000020/0001_01.png'  
  file_contents = tf.read_file(filename)  
    # image = tf.image.decode_image(file_contents, 3)  
  image = tf.image.decode_png(file_contents, 3)  
    rawimage = tf.identity(image)  
  
    # 随机裁剪  
  random_crop = tf.random_crop(image, image_size + (3,))  
  
    # 如果图像宽高小于目标宽高，则使用中心裁剪至目标宽高  
  # 如果图像宽高大于目标宽高，则pad 0 至目标宽高  
  resize_image_with_crop_or_pad = tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1])  
  
    # failed  
 # resize_with_crop_or_pad = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])  
 # 先对图像缩放到目标尺寸（保持宽高比），然后如果还是和目标尺寸不匹配则用0填充到目标宽高  
  resize_image_with_pad = tf.image.resize_image_with_pad(image, image_size[0], image_size[1])  
    resize_image_with_pad = tf.cast(resize_image_with_pad, dtype=tf.uint8)  
  
    # 中心裁剪  
  central_crop = tf.image.central_crop(image, 0.8)  
  
    # 先从左上角偏移一个offset作为左上角点，然后以offset_height + target_height, offset_width + target_width为右下角点  
  crop_to_bounding_box = tf.image.crop_to_bounding_box(image, 10, 20, image_size[0], image_size[1])  # target + offset <= image width&height, 目标图像宽高+offset之和一定要<=原始图像宽高。  
  
  # 先裁剪后缩放，目前感觉这个函数有点不大好用啊。  
  # crop_and_resize = tf.image.crop_and_resize(tf.expand_dims(image, axis=0), [[0.5, 0.6, 0.9, 0.8]], box_ind=[0], crop_size=(300, 400))  # [1]  
  crop_and_resize = tf.image.crop_and_resize(tf.expand_dims(image, axis=0), [[0.5, 0.6, 0.9, 0.8], [0.2, 0.6, 1.3, 0.9]], box_ind=[0, 0], crop_size=(300, 400))  
    crop_and_resize = tf.cast(crop_and_resize, dtype=tf.uint8)  
  
    # 在图片外围填充0  
  pad_to_bounding_box = tf.image.pad_to_bounding_box(image, offset_height=10, offset_width=10, target_height=200, target_width=200)  
  
    init = tf.initialize_all_variables()  
  
    with tf.Session() as sess:  
        sess.run(init)  
        cv2.imshow('rawimg', sess.run(rawimage))  
        cv2.imshow('random_crop1', sess.run(random_crop))  
        #cv2.imshow('random_crop2', sess.run(random_crop))  
 #cv2.imshow('random_crop3', sess.run(random_crop))  cv2.imshow('resize_image_with_crop_or_pad1', sess.run(resize_image_with_crop_or_pad))  
        cv2.imshow('resize_image_with_pad1', sess.run(resize_image_with_pad))  
        cv2.imshow('central_crop', sess.run(central_crop))  
        cv2.imshow('crop_to_bounding_box', sess.run(crop_to_bounding_box))  
        cv2.imshow('pad_to_bounding_box', sess.run(pad_to_bounding_box))  
  
        images = sess.run(crop_and_resize)  
        for i, image in enumerate(images):  
            cv2.imshow('crop_and_resize' + str(i), image)  
  
        cv2.waitKey(1000000)
```






