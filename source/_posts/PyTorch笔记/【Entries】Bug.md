---
title: 
date: 2020-7-2
tags:
categories: ["深度学习笔记"]
mathjax: true
---

the fatal while debug pytorch-pix2pixHD with pycharm:

```bash
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1579022060824/work/aten/src/THC/THCCachingHostAllocator.cpp line=278 error=700 : an illegal memory access was encountered
```

解决办法：
代码全局处 添加如下代码：
```python
torch.backends.cudnn.benchmark = False
# 实际上 torch.backends.cudnn.benchmark 默认就是 False
```

关于 torch.backends.cudnn.benchmark 的作用可参考文献 [1]






# 参考文献
[1] [torch.backends.cudnn.benchmark ?!](https://zhuanlan.zhihu.com/p/73711222)



