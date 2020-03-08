---
title: 
date: 2019-09-11
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述torch.autograd和backward和grad方法，以及关于autograd的一些小贴士
<!-- more -->

# torch.autograd.backward
```python
torch.autograd.backward(tensors,
                        grad_tensors=None,
                        ratain_graph=None,
                        crate_graph=False,
                        grad_variables=None)
```
**功能**：自动求取梯度

- **tensors**: 用于求导的张量，如loss，或者$y=wx+b$中的$y$. （有的时候tensor会直接调用自己的backward方法，这实际上tensor是在自己的backward方法中还是调用的 torch.autograd.backward ）
- **grad_tensors**: 多梯度权重，当有多个loss需要计算梯度的时候，这时候就需要设置各个loss之间权重的比例。
- **create_graph**: 创建导数计算图，用于高阶求导。create_graph设置为True时，表示创建导数的计算图，只有创建了导数的计算图之后，才能实现对导数的求导，即高阶导数。
- **retain_graph**: 保存计算图，由于pytorch采用的是动态图机制，pytorch在每一次反向传播结束后计算图都会被释放掉。所以如果想要继续使用计算图的话就要设置该参数为True.


## 代码示例1：演示backward的retain_graph参数的作用
```python
import torch
torch.manual_seed(10)

flag = True
# flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)  # 如果这里的retain_graph不设置为True，则y下面再次执行backward时会报错
    print(w.grad)
    y.backward()


# output:
tensor([5.])
```

## 代码示例2：演示backward参数中grad_tensors的使用
```python
flag = True
# flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)    dy0/dw = 5
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的 grad_tensors

    print(w.grad)


# output:
tensor([9.])
```
$$
\begin{aligned}
    \frac{\partial y_0}{\partial w} &= 5 \\
    \frac{\partial y_1}{\partial w} &= 2 \\
    \frac{\partial loss}{\partial w} &= grad\_tensors[0] \cdot \frac{\partial y_0}{\partial w} + grad\_tensors[1] \cdot \frac{\partial y_1}{\partial w} \\
    &= 1 \cdot 5 + 2 \cdot 2 = 9
\end{aligned}
$$



# torch.autograd.grad
```python
torch.autograd.grad(outputs, 
                    inputs, 
                    grad_outputs=None, 
                    retain_graph=None, 
                    create_graph=False, 
                    only_inputs=True, 
                    allow_unused=False)
```
**功能**：求取梯度

- **outputs**: 用于求导的张量，如loss，同torch.autograd.backward 中的tensors参数。
- **inputs**: 需要梯度的张量，例如 $y = wx + b$ 中的 $w$
- **create_graph**: 同torch.autograd.backward
- **retain_graph**: 同torch.autograd.backward
- **grad_outputs**：多梯度权重，同torch.autograd.backward的grad_tensors参数

## 代码示例：演示grad中的create_graph参数的作用
```python
flag = True
# flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)
```

# autograd小贴士
## 梯度不会自动清零
```python
flag = True
# flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)
        
        # 如果梯度不清零，则w.grad会不断累加
        # w.grad.zero_()  # 下划线表示 in-place 操作，即原地操作


# output:
tensor([5.])
tensor([10.])
tensor([15.])
tensor([20.])
```


## 依赖于叶子节点的节点，requires_grad默认为True
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/computational_graph.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp;  计算图</center>
如图1中的 节点a 就是依赖于子节点的节点，a 的 requires_grad 默认会被设置为True.

```python
flag = True
# flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)
```


## 叶子节点不可执行 in-place 操作

所谓 in-place 操作就是就地改变variable中的值的操作，下面举例说明。

**非in-place操作**
```python
a = torch.ones((1, ))
print(id(a), a)

a = a + torch.ones((1, ))
print(id(a), a)


# output:
140248928586416 tensor([1.])
140248828188496 tensor([2.])
```


**in-place操作**
```python
a = torch.ones((1, ))
print(id(a), a)

a += torch.ones((1, ))
print(id(a), a)


# output:
139857246825064 tensor([1.])
139857246825064 tensor([2.])
```

> **叶子节点不可执行 in-place 操作的原因**: 前向传播是记录的 w 的地址，在反向传播时如果还要读取 w 的数据，则只能根据 w 的地址取读取 w 的数据，而如果执行了 in-place 操作，改变了该地址中的数据，那么在反向传播的时候求取的梯度就不对了，所以不允许对叶子节点执行 in-place 操作。

```python
flag = True
# flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)

    y.backward()


# output:
  File "/path/to/dev/proml/tutorial/autograd.py", line 125, in <module>
    w.add_(1)
RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
```
上面的代码运行时会报错，因为其尝试对一个叶子节点做 in-place 操作。



# 参考文献
[1] DeepShare.net > PyTorch框架