---
title: 
date: 2019-09-11
tags:
categories: ["PyTorch笔记"]
mathjax: true
---


```python
torch.cross(input, other, dim=-1, out=None)  #叉乘(外积)

torch.dot(tensor1, tensor2)  #返回tensor1和tensor2的点乘

torch.mm(mat1, mat2, out=None) #返回矩阵mat1和mat2的乘积
torch.matmul(mat1, mat2, out=None)  # 同torch.mm
# 对矩阵`mat1`和`mat2`进行相乘。 如果`mat1` 是一个n×m张量，`mat2` 是一个 m×p 张量，将会输出一个 n×p 张量`out`。

torch.eig(a, eigenvectors=False, out=None) #返回矩阵a的特征值/特征向量 

torch.det(A)  #返回矩阵A的行列式

torch.trace(input) #返回2-d 矩阵的迹(对对角元素求和)

torch.diag(input, diagonal=0, out=None) #

torch.histc(input, bins=100, min=0, max=0, out=None) #计算input的直方图

torch.tril(input, diagonal=0, out=None)  #返回矩阵的下三角矩阵，其他为0

torch.triu(input, diagonal=0, out=None) #返回矩阵的上三角矩阵，其他为0
```


# 参考文献
[1] [pytorch入坑一 | Tensor及其基本操作](https://zhuanlan.zhihu.com/p/36233589)

