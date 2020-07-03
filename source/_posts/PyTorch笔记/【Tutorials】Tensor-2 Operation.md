---
title: 
date: 2019-09-11
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

本篇主要介绍tensor的拼接、切分、索引、变换、数学运算。
<!-- more -->


# Tensor拼接
## torch.cat
```python
torch.cat(tensors,
          dim=0,
          out=None)
```

**功能**：将tensor按维度进行拼接，不会扩张tensor的维度
- **tensors**: tensor序列
- **dim**: 要拼接的维度

**代码示例**:
```python
import torch
torch.manual_seed(1)

flag = True
# flag = False
if flag:
    t = torch.ones((2, 3))

    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t, t], dim=1)

    print("t_0:{}\nshape:{}\n\nt_1:{}\nshape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# 输出结果如下：
t_0:tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
shape:torch.Size([4, 3])

t_1:tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
shape:torch.Size([2, 9])
```

## torch.stack
```python
torch.stack(tensors,
            dim=0,
            out=None)
```

**功能**：在新创建的维度上拼接，与torch.cat不同的是 torch.stack 会扩张tensor的维度
- **tensors**: tensor序列
- **dim**: 要拼接的维度

```python
flag = True
# flag = False
if flag:
    # t = torch.ones((2, 3))
    t = torch.arange(0, 6).reshape((2, 3))

    t0_stack = torch.stack([t, t], dim=0)  # 已经有第0个维度还要在dim=0的维度上拼接，那就需要把原有的维度都整体向后挪一位
    t1_stack = torch.stack([t, t], dim=1)
    t2_stack = torch.stack([t, t], dim=2)

    print("\nt0_stack:{}\nshape:{}\n".format(t0_stack, t0_stack.shape))
    print("\nt1_stack:{}\nshape:{}\n".format(t1_stack, t1_stack.shape))
    print("\nt2_stack:{}\nshape:{}\n".format(t2_stack, t2_stack.shape))


# output:
t0_stack:tensor([[[0, 1, 2],
                  [3, 4, 5]],

                 [[0, 1, 2],
                  [3, 4, 5]]])
shape:torch.Size([2, 2, 3])

t1_stack:tensor([[[0, 1, 2],
                  [0, 1, 2]],

                 [[3, 4, 5],
                  [3, 4, 5]]])
shape:torch.Size([2, 2, 3])

t2_stack:tensor([[[0, 0],
                  [1, 1],
                  [2, 2]],

                 [[3, 3],
                  [4, 4],
                  [5, 5]]])
shape:torch.Size([2, 3, 2])
```


# Tensor切分
## torch.split
```python
torch.chunk(input,
            chunks,
            dim=0)
```

**功能**：将张量按维度dim进行平均切分

- **input**: 要切分的tensor
- **chunks**: 要切分的份数，**注意: 如果不能整除，最后一份tensor长度要小于其他tensor的**
- **dim**: 要切分的维度

**返回值**：tensor-list，len(tensor-list) == input.shape[dim] / chunks 向上取整，例如7/3=2.3，那么此时的len(tensor-list)就等于3.

```python
flag = True
# flag = False
if flag:
    a = torch.ones((2, 7))  # 7
    list_of_tensors = torch.chunk(a, dim=1, chunks=3)   # 3

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))


# output:
第1个张量：tensor([[1., 1., 1.],
                 [1., 1., 1.]]), shape is torch.Size([2, 3])
                 
第2个张量：tensor([[1., 1., 1.],
                 [1., 1., 1.]]), shape is torch.Size([2, 3])

第3个张量：tensor([[1.],
                 [1.]]), shape is torch.Size([2, 1])
```


## torch.split
```python
torch.split(input,
            split_size_or_sections,
            dim=0)
```

**功能**：将张量按维度dim切分，与torch.chunks不同的是，torch.split可以指定切分后每一份的长度。

- **input**: 要切分的tensor
- **split_size_or_sections**: 为 int 时，表示每一份的长度；为list时，表示按list元素切分，此时list中的元素之和应该等于input.shape[dim]，否则会报错。
- **dim**: 要切分的维度

**返回值**：tensor-list

```python
flag = True
# flag = False
if flag:
    t = torch.ones((2, 5))

    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)  # [2 , 1, 2]
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))

    # list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
    # for idx, t in enumerate(list_of_tensors):
    #     print("第{}个张量：{}, shape is {}".format(idx, t, t.shape))


# outpu:
第1个张量：tensor([[1., 1.],
                 [1., 1.]]), shape is torch.Size([2, 2])

第2个张量：tensor([[1.],
                 [1.]]), shape is torch.Size([2, 1])

第3个张量：tensor([[1., 1.],
                 [1., 1.]]), shape is torch.Size([2, 2])
```


# tensor 索引
## torch.index_select
```python
torch.index_select(input,
                   dim=0,
                   index,
                   out=None)
```

**功能**：在维度dim上，按index索引数据

- **input**: 要索引的tensor
- **dim**: 要索引的维度
- **index**: 要索引数据的序号，注意index的数据类型必须是long型

**返回值**：依index索引数据拼接的张量

```python
flag = True
# flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)    # index的数据类型必须是long, 如果是float则会报错
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))


# output:
t:
tensor([[4, 5, 0],
        [5, 7, 1],
        [2, 5, 8]])
t_select:
tensor([[4, 5, 0],
        [2, 5, 8]])
```


## torch.masked_select
```python
torch.masked_select(input,
                    mask,
                    out=None)
```

**功能**：按mask中的True进行索引，通常该方法用来筛选数据。

- **input**: 要索引的tensor
- **mask**: 与input同shape的布尔型张量

**返回值**：一维张量

| tensor function |  |  |
| --- | --- | --- |
| tensor.ge | greater than or equal | ≥ |
| tensor.gt | greater than  | ＞ |
| tensor.le | less than or equal | ≤ |
| tensor.lt | less than | ＜ |

```python
flag = True
# flag = False

if flag:

    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)  # ge is mean greater than or equal/   gt: greater than  le  lt
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))


# output:
t:
tensor([[4, 5, 0],
        [5, 7, 1],
        [2, 5, 8]])

mask:
tensor([[ True,  True,  True],
        [ True, False,  True],
        [ True,  True, False]])

t_select:
tensor([4, 5, 0, 5, 1, 2, 5]) 
```


# Tensor 变换
## torch.reshape
```python
torch.reshape(input,
              shape)
```

**功能**：变换tensor shape.

**注意：当张量在内存中是连续时，新张量与input共享数据内存**

- **input**: 要变换的tensor
- **shape**: 新张量的shape

```python
flag = True
# flag = False

if flag:
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (-1, 2, 2))    # -1
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))

    t[0] = 1024
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    print("t.data 内存地址:{}".format(id(t.data)))
    print("t_reshape.data 内存地址:{}".format(id(t_reshape.data)))


# output:
t:tensor([5, 4, 2, 6, 7, 3, 1, 0])

t_reshape:
tensor([[[5, 4],
         [2, 6]],

        [[7, 3],
         [1, 0]]])

t:tensor([1024,    4,    2,    6,    7,    3,    1,    0])

t_reshape:
tensor([[[1024,    4],
         [   2,    6]],

        [[   7,    3],
         [   1,    0]]])

t.data 内存地址:140155517373824
t_reshape.data 内存地址:140155517373536
```


## torch.transpose
```python
torch.transpose(input,
                dim0,
                dim1)
```

**功能**：变换tensor的两个维度，在图像的预处理中经常会使用到，例如将输入shape为(c,h,w)的图像变换为shape为(h,w,c)的图像的做法就是：先交换 c、h 的位置得到 (h,c,w) 再交换 c、w 的位置得到 (h,w,c)

- **input**: 要变换的tensor
- **dim0**: 要变换的维度
- **dim1**: 要变换的维度


## torch.t

```python
torch.t(input)
```
**功能**：2维张量转.对矩阵而言，等价于 torch.transpose(input, 0, 1)

- **input**: 要变换的tensor


## torch.squeeze
```python
torch.squeeze(input,
              dim=None,
              out=None)
```
**功能**：压缩长度为1的维度(轴)

- **input**: 要变换的tensor
- **dim**: 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为1时，可以被移除

```python
flag = True
# flag = False

if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq_dimNone = torch.squeeze(t)
    t_sq_dim0 = torch.squeeze(t, dim=0)
    t_sq_dim1 = torch.squeeze(t, dim=1)
    print('t:\n{}'.format(t))
    print('t.shape:\n{}'.format(t.shape))
    print('t_sq_dimNone.shape:\n{}'.format(t_sq_dimNone.shape))
    print('t_sq_dim0.shape:\n{}'.format(t_sq_dim0.shape))
    print('t_sq_dim1.shape:\n{}'.format(t_sq_dim1.shape))


# output:
t:tensor([[[[0.7576],
            [0.2793],
            [0.4031]],

           [[0.7347],
            [0.0293],
            [0.7999]]]])

t.shape:
torch.Size([1, 2, 3, 1])

t_sq_dimNone.shape:
torch.Size([2, 3])

t_sq_dim0.shape:
torch.Size([2, 3, 1])

t_sq_dim1.shape:
torch.Size([1, 2, 3, 1])
```

## torch.unsqueeze

```python
torch.unsqueeze(input,
                dim,
                out=None)
```
**功能**：依据dim扩展维度

- **input**: 要变换的tensor
- **dim**: 要扩展的维度


# tensor数学运算
## 加减乘除
```python
torch.add()
torch.addcdiv()
torch.addcmul()
torch.sub()
torch.div()
torch.mul()
```

下面举几个例子吧：

### torch.add
```python
torch.add(input,
          alpha=1,
          other,
          out=None)
```

**功能**：逐元素计算 input + alpha × other
- **input**: 第一个张量
- **input**: 乘项因子
- **other**: 第二个张量

```python
flag = True
# flag = False

if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)

    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))


# output:
t_0:
tensor([[ 0.6614,  0.2669,  0.0617],
        [ 0.6213, -0.4519, -0.1661],
        [-1.5228,  0.3817, -1.0276]])

t_1:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])

t_add_10:
tensor([[10.6614, 10.2669, 10.0617],
        [10.6213,  9.5481,  9.8339],
        [ 8.4772, 10.3817,  8.9724]])
```

### torch.addcmul
add combine mul，即加法结合乘法

```python
torch.addcmul(input,
              value=1,
              tensor1,
              tensor2,
              out=None)
```

**功能**：$out_i = input_i + value \times tensor1_i \times tensor2_i$


### torch.addcdiv
add combine div，即加法结合除法
```python
torch.addcdiv(input,
              value=1,
              tensor1,
              tensor2,
              out=None)
```
**功能**：$out_i = input_i + value \times \frac{tensor1_i}{tensor2_i}$



## 对数,指数,幂函数
```python
torch.log(input, out=None)
torch.log10(input, out=None)
torch.log2(input, out=None)
torch.exp(input, out=None)
torch.pow()
```

## 三角函数
```python
torch.abs(input, out=None)
torch.acos(input, out=None)
torch.cosh(input, out=None)
torch.cos(input, out=None)
torch.asin(input, out=None)
torch.atan(input, out=None)
torch.atan2(input, other, out=None)
```


## 数值裁剪

```python
    def clamp(self, min, max，out=None): # real signature unknown; restored from __doc__
        """
        clamp_(min, max) -> Tensor
        
        In-place version of :meth:`~Tensor.clamp`
        """
        pass
```

计算公式如下：
$$
y_i = 
\begin{cases}
    \text{min, if } x_i < \text{min} \\
    x_i, \text{if min } <= x_i <= \text{max} \\
    \text{max, if } x_i > \text{max}
\end{cases}
$$


## 逻辑运算

参考文献[2]
```python
torch.logical_xor

torch.logical_not

torch.logical_not

# 似乎pytorch有些版本不支持逻辑或
torch.logical_or
```





# 参考文献
[1] DeepShare.net > PyTorch框架 \
[2] [Pytorch官方文档](https://pytorch.org/docs/stable/torch.html?highlight=logical#torch.logical_xor)