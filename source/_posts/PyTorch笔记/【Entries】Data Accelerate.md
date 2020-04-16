---
title: 
date: 2020-04-10
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本文主要记录PyTorch IO提速加速训练。
<!-- more -->
在开发中如果训练的数据集是由大量小文件组成的，由于内存限制很难一次将所有数据都加载到内存中，所以不得不在训练中实时加载数据，然后这么多的小文件读取会受制于系统IO瓶颈，于是就有了这篇文章来总结训练中的IO提速方案。亲测过 apex、lmdb，目前来看lmdb效果较apex好，prefetch_generator的BackgroundGenerator也很管用，其他加速手段待尝试。

> 目前亲测过 apex、lmdb，确有提速，但还是没有能完全解决训练数据加载卡顿问题，只是相对什么都不用效果要好而已。


# NVIDIA apex
见参考文献

# NVIDIA dali
见参考文献

# 将原始图片存成一个大文件

Caffe 在图像分类模型的训练时, 效率起见, 未直接从图片列表读取图片, 训练数据往往是采用 LMDB 或 HDF5 格式 [4].

LMDB格式的优点：
- 基于文件映射IO（memory-mapped），数据速率更好
- 对大规模数据集更有效.

HDF5的特点：
- 易于读取
- 类似于mat数据，但数据压缩性能更强
- 需要全部读进内存里，故HDF5文件大小不能超过内存，可以分成多个HDF5文件，将HDF5子文件路径写入txt中.
- I/O速率不如LMDB.

## lmdb
我目前用的是这种，参考文献[1,2,3]，待整理。。。

## 其他各种文件格式
参考文献里很多文章已经提到了，时间仓促这里就不再赘述了。


# 参考文献

[1] [PyTorch使用LMDB数据库加速文件读取](https://www.yuque.com/lart/ugkv9f/hbnym1)
[2] [Caffe - 基于 Python 创建LMDB/HDF5格式数据](https://www.aiuai.cn/aifarm67.html)
[3] [Efficient-PyTorch#data-loader](https://github.com/Lyken17/Efficient-PyTorch#data-loader) (我的初版代码参考此文)
[4] [PyTorchTricks](https://github.com/lartpang/PyTorchTricks)
[基于文件存储UFS的Pytorch训练IO优化实践](https://zhuanlan.zhihu.com/p/115507582) (定制付费方案，暂不考虑)
[5] [nvidia-dali-for-pytorch](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/plugins/pytorch_tutorials.html)

[nvidia-dali](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html#Pipeline)

[Pytorch IO提速](https://www.cnblogs.com/king-lps/p/10936374.html)
[PyTorch加速数据读取](https://tianws.github.io/skill/2019/08/27/gpu-volatile/)
[优化Pytorch的数据加载](https://doc.flyai.com/blog/improve_dataload_pytorch.html)
[pytorch加速加载方案](https://www.cnblogs.com/zhengmeisong/p/11995374.html)
