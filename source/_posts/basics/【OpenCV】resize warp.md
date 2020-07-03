---
title: 
date: 2020-6-30
tags:
categories: ["深度学习笔记"]
mathjax: true
---
<!--more-->


# 图像翻转

左右翻转：
```python
srcimg == cv2.imread('/path/to/your/image.jpg')
lf_flip = srcimg[:, ::-1, :]
```

# 图像 resize 变换

## 图像、坐标缩放
```python
import cv2
import numpy as np


if __name__ == '__main__':  # 图像缩放，坐标缩放
    srcimg = cv2.imread("/home/mtbase/tangni/res/lena.png", 0)  # 读取灰度图
    h, w = srcimg.shape
    resize_ratio = (0.5, 0.5)  # 定义缩放比例(ratio_h, ratio_w)
    nh, nw = int(h*resize_ratio[0]), int(w*resize_ratio[1])  # 缩放后的宽高
    M = np.mat([[resize_ratio[0], 0], [0, resize_ratio[1]]])  # 缩放矩阵
    resize_img = np.zeros((nw, nh))  # 用于存储缩放后的图像
    points = np.array([(100, 150), (200, 250)], dtype=np.float64)  # 位于原图上的点

    # 坐标点缩放
    new_points = np.dot(points, M)  # 位于缩放后的图像上的点

    # 图像缩放
    for r in range(nh):
        for l in range(nw):
            v = np.dot(M.I, np.array([r, l]).T)
            resize_img[r, l] = srcimg[int(v[0, 0]), int(v[0, 1])]

    # 图像缩放也可以用 opencv 的借口 resize 来做。
    # resize_img = cv2.resize(srcimg, (nw, nh))

    for p in points.astype(np.int32):
        cv2.circle(srcimg, (p[0], p[1]), 2, (255, 0, 0), 1)

    for p in new_points.astype(np.int32):
        cv2.circle(resize_img, (p[0], p[1]), 2, (255, 0, 0), 1)

    cv2.imshow("srcimg", srcimg)
    cv2.imshow("resize_img", resize_img.astype("uint8"))
    cv2.waitKey()
```


# 图像 warp 变换
```python
def warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None): # real signature unknown; restored from __doc__
    """
    warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
    .   @brief Applies an affine transformation to an image.
    .   
    .   The function warpAffine transforms the source image using the specified matrix:
    .   
    .   \f[\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\f]
    .   
    .   when the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
    .   with #invertAffineTransform and then put in the formula above instead of M. The function cannot
    .   operate in-place.
    .   
    .   @param src input image.
    .   @param dst output image that has the size dsize and the same type as src .
    .   @param M \f$2\times 3\f$ transformation matrix.
    .   @param dsize size of the output image.
    .   @param flags combination of interpolation methods (see #InterpolationFlags) and the optional
    .   flag #WARP_INVERSE_MAP that means that M is the inverse transformation (
    .   \f$\texttt{dst}\rightarrow\texttt{src}\f$ ).
    .   @param borderMode pixel extrapolation method (see #BorderTypes); when
    .   borderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
    .   the "outliers" in the source image are not modified by the function.
    .   @param borderValue value used in case of a constant border; by default, it is 0.
    .   
    .   @sa  warpPerspective, resize, remap, getRectSubPix, transform
    """
    pass
```
参数说明：
- **src**: 
- **M**: 旋转矩阵
- **dsize**: 目标尺寸
- **dst**: 目标存储图片，若空则直接返回目标图片
- **flags**: interpolation(插值)方法和 WARP_INVERSE_MAP 选项的组合，目前主要用来指定插值方法
- **borderMode**: 边界像素模式
- **borderValue**: 边界填充值，默认为0


## 图像、坐标旋转
```python
if __name__ == '__main__':  # 图像、坐标旋转
    img = np.ones((512, 800, 3), dtype=np.uint8) * 128
    img_h, img_w, _ = img.shape
    angle = 10
    cw, ch = img_w // 2, img_h // 2
    rect_size = (360, 200)
    points = np.array([(cw - rect_size[0]/2, ch - rect_size[1]/2), (cw + rect_size[0]/2, ch + rect_size[1]/2)])
    p = points.astype(np.int)
    cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 255))
    cv2.imshow('src_img', img)
    cv2.waitKey()

    M = cv2.getRotationMatrix2D((cw, ch), angle, 1.0)

    # 对图像的旋转
    # img = cv2.warpAffine(img, M, (img_w, img_h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)

    # 对坐标的旋转
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    points = M.dot(points_ones.T).T
    p = points.astype(np.int)

    cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 255, 0))
    cv2.imshow('rotate_img', img)
    cv2.waitKey()
```


## 图像旋转黑边裁剪

```python
def calc_rotate_offset(angle, rect_size):
    """计算因旋转导致的边界黑边偏移
    :param angle:
    :param rect_size: rect_size == (width, height)
    :return:
    """
    tana = math.tan(angle * math.pi / 180.0)
    sina = math.sin(angle * math.pi / 180.0)
    A = 1.0 / tana + 1.0 / sina

    P = (rect_size[1] - A * rect_size[0]) / (tana - A)
    Q = A / (tana - A)

    B = 1 + tana ** 2
    delta = (2 * P * Q * B) ** 2 - 4 * (B - 1) * B * (P ** 2)

    # 得到两个解：
    # c1 = (-2 * P * Q * B + np.sqrt(delta)) / (2 * (B - 1))
    c = (-2 * P * Q * B - np.sqrt(delta)) / (2 * (B - 1))

    a = P + Q * c
    b = a * tana
    d = rect_size[0] - a - c
    print(a, b, c, d)

    return int(b+1), int(d+1)


import math

if __name__ == '__main__':  # 图像旋转后会有黑边，本实验探索在旋转后的图片中裁剪出不包含黑边的最大区域。
    img = np.zeros((512, 800, 3), dtype=np.uint8)
    img_h, img_w, _ = img.shape
    angle = 10

    cw, ch = img_w // 2, img_h // 2
    rect_w, rect_h = 360, 200
    rect_size = (rect_w, rect_h)
    points = np.array(
        [(cw - rect_size[0] / 2, ch - rect_size[1] / 2), (cw + rect_size[0] / 2, ch + rect_size[1] / 2)])
    p = points.astype(np.int)
    cv2.rectangle(img, tuple(p[0]), tuple(p[1]), (0, 0, 255))

    while True:  # 早期实验代码，最后总结如上面封装好的函数 calc_rotate_offset
        for angle in [10, 20, 30, 40, 50, 60, 70, -10, -20, -30, -40, -50, -60, -70]:
            tana = math.tan(angle * math.pi / 180.0)
            sina = math.sin(angle * math.pi / 180.0)
            A = 1.0/tana + 1.0/sina  # TODO:

            P = (rect_h - A * rect_w) / (tana - A)
            Q = A / (tana - A)

            B = 1 + tana ** 2
            delta = (2*P*Q*B)**2 - 4*(B-1)*B*(P**2)

            c1 = (-2*P*Q*B + np.sqrt(delta))/(2*(B-1))
            c = (-2*P*Q*B - np.sqrt(delta))/(2*(B-1))

            a = P + Q*c
            b = a * tana
            d = rect_w - a - c
            print(a, b, c, d)

            M = cv2.getRotationMatrix2D((cw, ch), angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            imag = img + rotated_img

            cv2.imshow('rotate_img', imag)
            cv2.waitKey()
```


## 图像旋转采用不同插值算法所造成的损失比较

实测最近邻插值在图像旋转时造成的损失最小。

```python
import cv2


if __name__ == '__main__':  # 使用较小的图像矩阵，测试图像旋转后的数值变化
    img = np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35],
                    [40, 41, 42, 43, 44, 45], [50, 51, 52, 53, 54, 55]], dtype=np.float64)
    img = np.array([img, img, img])
    img = np.reshape(img, (6, 6, 3))
    ang = 1  # 旋转角度
    nw, nh, _ = img.shape
    cx, cy = nw // 2, nh // 2
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)

    if False:  # 参考文献[1]
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        # compute the new bounding dimensions of the image
        nw = int((nh * sin) + (nw * cos))
        nh = int((nh * cos) + (nw * sin))
        # adjust the rotation matrix to take into account translation
        m[0, 2] += (nw / 2) - cx
        m[1, 2] += (nh / 2) - cy

    print('srcimg: \n', img[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT)
    print('default interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
    print('linear interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    print('nearest interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_CUBIC)
    print('cubic interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_AREA)
    print('area interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LANCZOS4)
    print('lanczos4 interp: \n', wimg[:, :, 0])

    cv2.imshow('show', wimg)
    cv2.waitKey()

# output:

# srcimg: 
 [[ 0.  3. 10. 13. 20. 23.]
 [30. 33. 40. 43. 50. 53.]
 [ 0.  3. 10. 13. 20. 23.]
 [30. 33. 40. 43. 50. 53.]
 [ 0.  3. 10. 13. 20. 23.]
 [30. 33. 40. 43. 50. 53.]]

# default interp: 
 [[ 0.1875   3.4375  10.1875  13.4375  21.125   23.9375 ]
 [28.21875 32.28125 39.15625 43.21875 49.15625 52.0625 ]
 [ 1.96875  4.15625 11.03125 13.21875 21.03125 23.9375 ]
 [28.125   32.0625  39.0625  43.      49.0625  52.0625 ]
 [ 1.875    3.84375 10.71875 12.90625 20.71875 23.84375]
 [28.125   31.96875 38.84375 42.90625 49.78125 52.90625]]

# linear interp: 
 [[ 0.1875   3.4375  10.1875  13.4375  21.125   23.9375 ]
 [28.21875 32.28125 39.15625 43.21875 49.15625 52.0625 ]
 [ 1.96875  4.15625 11.03125 13.21875 21.03125 23.9375 ]
 [28.125   32.0625  39.0625  43.      49.0625  52.0625 ]
 [ 1.875    3.84375 10.71875 12.90625 20.71875 23.84375]
 [28.125   31.96875 38.84375 42.90625 49.78125 52.90625]]

# nearest interp:  和原图像素值最为接近
 [[ 0.  3. 10. 13. 20. 23.]
 [30. 33. 40. 43. 50. 53.]
 [ 0.  3. 10. 13. 20. 23.]
 [30. 33. 40. 43. 50. 53.]
 [ 0.  3. 10. 13. 20. 23.]
 [30. 33. 40. 43. 50. 53.]]

# cubic interp: 
 [[-1.1885376   2.78198236  9.73706035 13.46313477 21.1833645  23.87776142]
 [29.81298008 33.16816671 40.15669202 43.23294067 50.14037262 52.98205503]
 [ 0.40441132  3.31900022 10.30752552 13.23294067 20.31249214 23.15417455]
 [29.66308594 32.91394043 39.91394043 43.         49.91394043 52.91394043]
 [ 0.26879883  2.85962676  9.8531188  12.77853394 19.83183267 22.99727605]
 [28.3590082  32.02764093 39.02113297 42.77853394 50.44821106 53.61365445]]

# area interp: 
 [[ 0.1875   3.4375  10.1875  13.4375  21.125   23.9375 ]
 [28.21875 32.28125 39.15625 43.21875 49.15625 52.0625 ]
 [ 1.96875  4.15625 11.03125 13.21875 21.03125 23.9375 ]
 [28.125   32.0625  39.0625  43.      49.0625  52.0625 ]
 [ 1.875    3.84375 10.71875 12.90625 20.71875 23.84375]
 [28.125   31.96875 38.84375 42.90625 49.78125 52.90625]]

# lanczos4 interp: 
 [[-2.31186262  2.21265227  9.0798498  13.32227143 21.65598169 24.30438353]
 [30.57821477 33.54302343 40.48430011 43.15772589 49.75035795 52.5718655 ]
 [ 0.10356931  3.18112859 10.12240528 13.15772456 20.36115352 23.18266107]
 [29.54831741 32.83990528 39.83990599 43.         50.0272637  53.02726384]
 [ 0.99350557  3.24964558 10.29297693 12.8503357  19.45698278 22.64000655]
 [27.40469293 31.53667515 38.5800065  42.85033876 51.00184913 54.1848729 ]]
```


# 参考文献
[1] [opencv 无损旋转后 坐标不知如何重新计算回来](https://bbs.csdn.net/topics/393122553?list=lz)


