---
title: 【PyTorch笔记】Tenosr
date: 2019-5-02
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

# 直接创建

## torch.tensor
```python
torch.tensor(data,
             dtype=None,
             device=None,
             requires_grad=False,
             pin_memory=False)
```

<!-- more -->

- **data**: 数据，可以是list, numpy
- **dtype**: 数据类型，默认与data的数据类型一致
- **device**: 所在设备，cuda/cpu
- **requires_grad**: 是否需要梯度
- **pin_memory**: 是否存于锁页内存，这与转换效率有关，通常设置成False即可

```python
import torch
import numpy as np

# 通过torch.tensor创建张量
arr = np.ones((3, 3))
print(arr)
print("ndarray的数据类型：", arr.dtype)
'''
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
ndarray的数据类型： float64
'''

t = torch.tensor(arr)  # 存储在cpu上
print(t)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
'''

# 将从arr创建tensor，并将其搬运到gpu上，注意这可能会耗费一些时间
t = torch.tensor(arr, device='cuda')
print(t)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], device='cuda:0', dtype=torch.float64)
'''
```


## torch.from_numpy
注意：从torch.from_numpy 创建的tensor与原ndarray**共享内存**，当修改其中一个的数据时，另一个也将会被改动

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.from_numpy(arr)
print("numpy array: ", arr)
print("ndarray的数据类型：", arr.dtype)
print("tensor : ", t)
print("tensor的数据类型：", t.dtype)
'''
numpy array:  [[1 2 3]
               [4 5 6]]
ndarray的数据类型： int64
tensor :  tensor([[1, 2, 3],
                  [4, 5, 6]])
tensor的数据类型： torch.int64
'''

# print("\n修改arr")
# arr[0, 0] = 0
# print("numpy array: ", arr)
# print("tensor : ", t)

print("\n修改tensor")
t[0, 0] = -1
print("numpy array: ", arr)
print("tensor : ", t)
'''
修改tensor
numpy array:  [[-1  2  3]
               [ 4  5  6]]
tensor :  tensor([[-1,  2,  3],
                  [ 4,  5,  6]])
'''

```


# 依据数值创建

## torch.zeros

```python
torch.zeros(size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```

功能：依size创建全 0 张量

- **size**: 张量的形状，如(3, 3), (3, 224, 224)
- **out**：输出张量，默认为空，如果指定则将结果拷贝到out并返回，也就是说此时的输出结果和out是同一个东西
- **dtype**: tensor type
- **layout**: 内存中的布局形式，有strided, sparse_coo等, 一般用strided，如果要创建稀疏张量的时候用sparse_coo可能会提高效率。
- **device**: 所在设备，gpu/cpu
- **requires_grad**: 是否需要梯度


```python
out_t = torch.tensor([0])  # 先随便创建一个tensor

t = torch.zeros((3, 3), out=out_t)

print('t: ', t)
print('out_t', out_t)
'''
t:  tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
out_t tensor([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])
'''

print('t内存地址：', id(t))
print('out_t内存地址：', id(out_t))
print(id(t) == id(out_t))
'''
t内存地址： 139742620961456
out_t内存地址： 139742620961456
True
'''

# 上例表面返回值t和out_t是同一个东西
```


## torch.zeros_like
```python
torch.zeros_like(input,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False)
```

功能：依input形状创建全 0 张量

- **input**: 必须是一个tensor, 不能是numpy.ndarray
- **dtype**: tensor type
- **layout**: 内存中的布局形式，有strided, sparse_coo等, 一般用strided，如果要创建稀疏张量的时候用sparse_coo可能会提高效率。
- **device**: 所在设备，gpu/cpu
- **requires_grad**: 是否需要梯度

```python
t1 = torch.zeros((3, 3))
t2 = torch.zeros_like(t1)
```


## torch.ones
```python
torch.ones(size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```


## torch.ones_like
```python
torch.ones_like(input,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False)
```


## torch.full

```python
torch.full(size,
            fill_value,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```

功能：依size创建张量，并往里面填充fill_value值

- **size**: 张量的形状，如(3, 3), (3, 224, 224)
- **fill_value**: 填充值
- **out**：输出张量，默认为空，如果指定则将结果拷贝到out并返回，也就是说此时的输出结果和out是同一个东西
- **dtype**: tensor type
- **layout**: 内存中的布局形式，有strided, sparse_coo等, 一般用strided，如果要创建稀疏张量的时候用sparse_coo可能会提高效率。
- **device**: 所在设备，gpu/cpu
- **requires_grad**: 是否需要梯度


```python
t = torch.full((3, 3), 2)
print(t, '\n', t.dtype)
'''
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]]) 
torch.float32
'''
```


## torch.full_like

```python
torch.full_like(input,
                 fill_value,
                 dtype=None,
                 layout=torch.strided,
                 device=None,
                 requires_grad=False)
```
功能：依input形状创建张量，并往里面填充fill_value值

- **input**: 必须是一个tensor, 不能是numpy.ndarray
- **fill_value**: 填充值
- **dtype**: tensor type
- **layout**: 内存中的布局形式，有strided, sparse_coo等, 一般用strided，如果要创建稀疏张量的时候用sparse_coo可能会提高效率。
- **device**: 所在设备，gpu/cpu
- **requires_grad**: 是否需要梯度


## troch.arange

```python
torch.arange(start,
             end,
             step=1,
             out=None,
             dtype=None,
             layout=torch.strided,
             device=None,
             requires_grad=False
             )
```

功能：创建等差的1维张量, 数值区间位于 [start, end)
- **start**: 数列起始值
- **end**: 数列结束值
- **step**: 数列公差，默认为1


```python
t = torch.arange(2, 10, 2)
print(t)
'''
tensor([2, 4, 6, 8])
'''
```


## torch.linspace

```python
torch.linspace(start,
             end,
             steps=100,
             out=None,
             dtype=None,
             layout=torch.strided,
             device=None,
             requires_grad=False
             )
```

功能：创建均分的1维张量, 数值区间位于 [start, end]

这个数列的步长：$\frac{end-start}{steps-1}$

- **start**: 数列起始值
- **end**: 数列结束值
- **steps**: 数列长度

```python
t = torch.linspace(2, 10, 5)
print(t)
'''
tensor([ 2.,  4.,  6.,  8., 10.])
'''
```


## torch.logspace

```python
torch.logspace(start,
               end,
               steps=100,
               base=10.0,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False
               )
```
功能：创建对数均分的1维张量

- **start**: 
- **end**: 
- **steps**: 数列长度
- **base**: 对数函数底，默认为10

功能：创建区间位于 $[base_{start}, base_{end}]$ 个数为steps的等比数列，

```python
t = torch.logspace(2, 10, 5, base=2)
print(t)
'''
tensor([   4.,   16.,   64.,  256., 1024.])
'''
```

将上述结果带入 $log_{base}x = log_{2}x$ 有： \
$[log_{2}4, \; log_{2}16, \; log_{2}64, \; log_{2}256, \; log_{2}1024] = [2, \; 4, \; 6, \; 8, \; 10]$

所以这个数列也称为对数均分数列。


## torch.eye
```python
torch.eye(n,
          m=None,
          out=None,
          dtype=None,
          layout=torch.strided,
          device=None,
          requires_grad=False
          )
```
功能：创建对数均分的1维张量

- **n**: 对角阵行数
- **m**: 对角阵列数，如果不指定则内部默认会等于n

功能：创建n行m列的单位对角阵（2维张量）


```python
t = torch.eye(2)
print(t)
'''
tensor([[1., 0.],
        [0., 1.]])
'''

t = torch.eye(2, 2)
print(t)
'''
tensor([[1., 0.],
        [0., 1.]])
'''

t = torch.eye(2, 3)
print(t)
'''
tensor([[1., 0., 0.],
        [0., 1., 0.]])
'''
```


# 依概率分布创建张量

## torch.normal

功能：从指定的均值为mean方差为std的正态分布中抽样。

torch.normal 有四种模式：\
mean为标量，std为标量 \
mean为标量，std为张量 \
mean为张量，std为标量 \
mean为张量，std为张量 \


```python
'''
"mean为标量，std为张量"
"mean为张量，std为标量"
"mean为张量，std为张量", 这3中情况的参数形式都差不多，但mean和std的具体参数类型不同。
'''
torch.normal(mean,
             std,
             out=None
             )
```

```python
'''
"mean为标量，std为标量" 这种情况需要指定size.
'''
torch.normal(mean,
             std,
             size,
             out=None
             )
```

上述是 torch.normal() 常用的主要参数，torch.normal() 完整参数列表如下：\
normal() received arguments expected one of: \
 * (Tensor mean, Tensor std, torch.Generator generator, Tensor out)
 * (Tensor mean, float std, torch.Generator generator, Tensor out)
 * (float mean, Tensor std, torch.Generator generator, Tensor out)
 * (float mean, float std, tuple of ints size, torch.Generator generator, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)


```python
# mean：张量 std: 张量
mean = torch.arange(1, 5, dtype=torch.float)
std = torch.arange(1, 5, dtype=torch.float)
t_normal = torch.normal(mean, std)
print("mean:{}\nstd:{}".format(mean, std))
print(t_normal)
'''
mean:tensor([1., 2., 3., 4.])
std:tensor([1., 2., 3., 4.])
tensor([ 1.3315,  3.4292, -2.2489,  7.2009])

在均值为mean[0]方差为std[0]的正态分布中抽样 得到normal[0]
……
在均值为mean[i]方差为std[i]的正态分布中抽样 得到normal[i]
'''


# mean：标量 std: 标量
t_normal = torch.normal(0., 1., size=(4,))
print(t_normal)
'''
tensor([ 0.4869, -0.7586, -0.4964, -0.7619])
'''


# mean：张量 std: 标量
mean = torch.arange(1, 5, dtype=torch.float)
std = 1
t_normal = torch.normal(mean, std)
print("mean:{}\nstd:{}".format(mean, std))
print(t_normal)
'''
mean:tensor([1., 2., 3., 4.])
std:1
tensor([2.6163, 1.4440, 2.0303, 3.0778])
'''
```


## torch.randn、torch.randn_like

```python
torch.randn(size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False
            )
```

功能：从标准正态分布中抽样


## torch.rand、torch.rand_like
```python
torch.rand(size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False
            )
```

功能：在区间[0, 1)的均匀分布上采样


## torch.randint、torch.randint_like
```python
torch.randint(low,
              high,
              size,
              out=None,
              dtype=None,
              layout=torch.strided,
              device=None,
              requires_grad=False
              )
```

功能：在区间为 [low, high) 的均匀分布上采样


## torch.randperm
```python
torch.rand(n,
            out=None,
            dtype=torch.int64,
            layout=torch.strided,
            device=None,
            requires_grad=False
            )
```

功能：生成从 0 到 n-1 的随机排列

- **n**: 张量长度
- **out**: 


## torch.bernoulli
```python
torch.bernoulli(input,
                *,
                generator=None,
                out=None
                )
```

功能：以 input 为概率，生成伯努利分布（0-1分布，两点分布）

- **input**: 概率值
- **out**: 


# 参考文献
[1] DeepShare.net
