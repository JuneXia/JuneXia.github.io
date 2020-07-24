---
title: 
date: 2020-6-30
tags:
categories: ["basics"]
mathjax: true
---
<!--more-->

# 常用图像保存格式
先简单介绍一下常用的图片格式以及他们的特点 [1]：

**BMP格式（无压缩）** \
位图（外语简称：BMP、外语全称：BitMaP）BMP是一种与硬件设备无关的图像文件格式，使用非常广。它采用位映射存储格式，除了图像深度可选以外，不采用其他任何压缩，因此，BMP文件所占用的空间很大。

**JPEG格式（有损压缩）** \
联合照片专家组（外语简称JPEG外语全称：Joint Photographic Expert Group）JPEG也是最常见的一种图像格式，它是由联合照片专家组（外语全称：Joint Photographic Experts Group），文件后辍名为"．jpg"或"．jpeg"，是最常用的图像文件格式，由一个软件开发联合会组织制定，是一种有损压缩格式，能够将图像压缩在很小的储存空间，图像中重复或不重要的资料会被丢失，因此容易造成图像数据的损伤。尤其是使用过高的压缩比例，将使最终解压缩后恢复的图像质量明显降低，如果追求高品质图像，不宜采用过高压缩比例。但是JPEG压缩技术十分先进，它用有损压缩方式去除冗余的图像数据，在获得极高的压缩率的同时能展现十分丰富生动的图像，换句话说，就是可以用最少的磁盘空间得到较好的图像品质。而且JPEG是一种很灵活的格式，具有调节图像质量的功能，允许用不同的压缩比例对文件进行压缩，支持多种压缩级别，压缩比率通常在10：1到40：1之间，压缩比越大，品质就越低；相反地，压缩比越小，品质就越好。

**PNG格式（无损压缩）** \
便携式网络图形（外语简称PNG、外语全称：Portable Network Graphics），是网上接受的最新图像文件格式。PNG能够提供长度比GIF小30%的无损压缩图像文件。它同时提供24位和48位真彩色图像支持以及其他诸多技术性支持。由于PNG非常新，所以并不是所有的程序都可以用它来存储图像文件，但Photoshop可以处理PNG图像文件，也可以用PNG图像文件格式存储。


# imwrite
```python
def imwrite(filename, img, params=None): # real signature unknown; restored from __doc__
    """
    imwrite(filename, img[, params]) -> retval
    .   @brief Saves an image to a specified file.
    .   
    .   The function imwrite saves the image to the specified file. The image format is chosen based on the
    .   filename extension (see cv::imread for the list of extensions). In general, only 8-bit
    .   single-channel or 3-channel (with 'BGR' channel order) images
    .   can be saved using this function, with these exceptions:
    .   
    .   - 16-bit unsigned (CV_16U) images can be saved in the case of PNG, JPEG 2000, and TIFF formats
    .   - 32-bit float (CV_32F) images can be saved in PFM, TIFF, OpenEXR, and Radiance HDR formats;
    .     3-channel (CV_32FC3) TIFF images will be saved using the LogLuv high dynamic range encoding
    .     (4 bytes per pixel)
    .   - PNG images with an alpha channel can be saved using this function. To do this, create
    .   8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels
    .   should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535 (see the code sample below).
    .   
    .   If the format, depth or channel order is different, use
    .   Mat::convertTo and cv::cvtColor to convert it before saving. Or, use the universal FileStorage I/O
    .   functions to save the image to XML or YAML format.
    .   
    .   The sample below shows how to create a BGRA image and save it to a PNG file. It also demonstrates how to set custom
    .   compression parameters:
    .   @include snippets/imgcodecs_imwrite.cpp
    .   @param filename Name of the file.
    .   @param img Image to be saved.
    .   @param params Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see cv::ImwriteFlags
    """
    pass
```
参数说明：
- **filename**: 
- **img**: 
- **params**: 指定保存格式选项

> 关于使用 imwrite 保存图片时的保存质量选择：
> - 对于JPEG格式的图片，这个参数表示从 0-100 的图片质量（CV_IMWRITE_JPEG_QUALITY）,默认值是95.
> - 对于PNG格式的图片，这个参数表示压缩级别（CV_IMWRITE_PNG_COMPRESSION）从0-9.较高的值意味着更小的尺寸和更长的压缩时间而默认值是3.
> - 对于PPM，PGM或PBM格式的图片，这个参数表示一个二进制格式标志（CV_IMWRITE_PXM_BINARY），取值为0或1，而默认值为1.
> - 其他参数并不重要，若感兴趣可参考文献[2,3]


代码示例：
```python
import cv2


if __name__ == '__main__':
    img_src = cv2.imread('xxxx.jpg')

    # 保存成jpg格式时，将IMWRITE_JPEG_QUALITY保存质量设置为100即是保存质量最高，其他参数作用不大
    cv2.imwrite('./test/src_img_quality100.jpg', img_src, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite('./test/src_img_quality100prog.jpg', img_src,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100, cv2.IMWRITE_JPEG_PROGRESSIVE, True])
    cv2.imwrite('./test/src_img_quality100prog_opt.jpg', img_src, [int(cv2.IMWRITE_JPEG_QUALITY), 100,
                                                                   cv2.IMWRITE_JPEG_PROGRESSIVE, True,
                                                                   cv2.IMWRITE_JPEG_OPTIMIZE, True])
    cv2.imwrite('./test/src_img_quality100prog_opt_jpeg2000-0.jpg', img_src, [int(cv2.IMWRITE_JPEG_QUALITY), 100,
                                                                              cv2.IMWRITE_JPEG_PROGRESSIVE, True,
                                                                              cv2.IMWRITE_JPEG_OPTIMIZE, True,
                                                                              cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,
                                                                              0])
    cv2.imwrite('./test/src_img_quality100prog_opt_jpeg2000-0_chroma100_.jpg', img_src,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100,
                 cv2.IMWRITE_JPEG_PROGRESSIVE, True,
                 cv2.IMWRITE_JPEG_OPTIMIZE, True,
                 cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 0,
                 cv2.IMWRITE_JPEG_CHROMA_QUALITY, 100])
    cv2.imwrite('./test/src_img_quality100prog_opt_jpeg2000-0_chroma100_luma100.jpg', img_src,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100,
                 cv2.IMWRITE_JPEG_PROGRESSIVE, True,
                 cv2.IMWRITE_JPEG_OPTIMIZE, True,
                 cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 0,
                 cv2.IMWRITE_JPEG_CHROMA_QUALITY, 100,
                 cv2.IMWRITE_JPEG_LUMA_QUALITY, 100])
    cv2.imwrite('./test/src_img_qualitynone.jpg', img_src)

    # 保存png格式，png是无损压缩，参数IMWRITE_PNG_COMPRESSION表示的是压缩级别，压缩级别越高，则压缩后所占用的磁盘空间越小，当然所需要的压缩时间也越长。
    # 注意：保存png格式时，参数IMWRITE_PNG_COMPRESSION控制的是压缩时间，对图片质量无影响，因为png格式就是无损压缩。
    cv2.imwrite('./test/src_img.png', img_src)
    cv2.imwrite('./test/src_img_cmpre0.png', img_src, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('./test/src_img_cmpre9.png', img_src, [cv2.IMWRITE_PNG_COMPRESSION, 9])
```


# 参考文献
[1] [使用imwrite调整保存的图片质量](https://blog.csdn.net/qq_33485434/article/details/79089069)
[2] [OpenCV笔记1：用imwrite函数来保存图片](http://blog.gqylpy.com/gqy/21819/)
[3] [opencv官方文档](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html)

