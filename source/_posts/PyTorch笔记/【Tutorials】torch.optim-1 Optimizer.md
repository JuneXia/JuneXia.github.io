---
title: 
date: 2020-02-15
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
有了 Loss 之后，怎样采用 Loss 去更新模型参数使得 Loss 逐步降低呢，这就是优化器（Optimizer）要干的活。
<!-- more -->

# 什么是优化器
&emsp; pytorch的优化器: **管理**并**更新**模型中可学习参数的值,使得模型输出更接近真实标签，其中“管理”指的是优化器可以修改哪一部分参数，而“更新”指的是优化器使用的某种策略去更新这些参数的值，而在神经网络中一般都是采用梯度下降更新策略去更新参数。

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer1.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer2.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图1 &nbsp; 左侧为一元函数，右侧为二元函数</center>
</html>

**导数**: 函数**在指定坐标轴上的**变化率 \
**方向导数**: 指定方向上的变化率，对于一元函数，通常不会考虑它的方向导数，因为它的方向导数很受限制；通常会在二元或者三元函数中去讨论方向导数的概念；\
**梯度**: 梯度是一个向量, 这个向量的方向为方向导数取得最大值的方向，而这个向量的模长就是方向导数的值（也就是变化率）。
**梯度下降**: 梯度是增长最快的方向，而梯度下降就是朝着梯度的负方向下降速度是最快的。

# Optimizer的属性
```python
class Optimizer(object):
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        # 伪代码
        param_groups = [{'params': param_groups}]
```
基本属性
- **defaults**: 优化器超参数，例如里面会存储学习率、momentum、weight_decay等超参数。
- **state**: 参数的缓存，如 momentum 的缓存（momentum会使用到前几次更新时的梯度）
- **params_groups**: 管理的参数组，params_groups是一个list，而这个list的每个元素又是一个dict，每个dict存储一组参数。具体可参见图2.
- **_step_count**: 记录更新次数，学习率调整中使用。


# Optimizer的方法

## .zero_grad
```python
class Optimizer(object):
    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
```
**功能**：清空所管理参数的梯度

**注意**：在Pytorch，张量梯度不自动清零，而是累加历史梯度，所以需要在每次使用梯度更新完参数后将参数的梯度清零，或者在backword之前将梯度清零也行。


代码示例：
```python
# ----------------------------------- zero_grad -----------------------------------
# flag = 0
flag = 1
if flag:
    print("weight.grad is {}\n".format(weight.grad))
    optimizer.zero_grad()
    print("after optimizer.zero_grad(), weight.grad is\n{}".format(weight.grad))
```


优化器中存储的参数的地址和参数自己的地址是一样的，这说明优化器中存储的只是参数的地址而已，这样可以解决内存空间。
```python
print("weight in optimizer:{}".format(id(optimizer.param_groups[0]['params'][0])))
print("weight in weight:{}".format(id(weight)))


# output:
weight in optimizer: 140320210504008
weight in weight:    140320210504008
```


## .step
```python
class Optimizer(object):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
```
**功能**：执行一步更新，具体更新方法会根据优化策略来进行。

代码示例：
```python
# -*- coding: utf-8 -*-
"""
# @file name  : optimizer_methods.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-10-14 10:08:00
# @brief      : optimizer's methods
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.optim as optim
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子

weight = torch.randn((2, 2), requires_grad=True)
weight.grad = torch.ones((2, 2))  # 为了便于观察效果这里将grad设置为1

optimizer = optim.SGD([weight], lr=0.1)

# ----------------------------------- step -----------------------------------
# flag = 0
flag = 1
if flag:
    print("weight before step:{}".format(weight.data))
    optimizer.step()        # 修改lr=1 0.1观察结果
    print("weight after step:{}".format(weight.data))


# output:
weight before step:
tensor([[0.6614, 0.2669],
        [0.0617, 0.6213]])

weight after step:
tensor([[ 0.5614,  0.1669],
        [-0.0383,  0.5213]])
```
计算过程：\
$w_{i} = w_{i-1} - lr \* \Delta = 0.6614 - 0.1 \* 1 = 0.5614$


## .add_param_group
```python
class Optimizer(object):
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
```
**功能**：添加参数组。具体来说就是：可以将参数分组添加到优化器，对不同组的参数有不同超参数设置，例如在模型的finetune当中可以将模型参数分成两组，让模型前面的特征提取部分的学习率小一些，而让后面我们自己添加的全连接层的学习率大一些。

代码示例：
```python
# ----------------------------------- add_param_group -----------------------------------
# flag = 0
flag = 1
if flag:
    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))

    w2 = torch.randn((3, 3), requires_grad=True)

    optimizer.add_param_group({"params": w2, 'lr': 0.0001})

    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))


# output:
optimizer.param_groups is
[
    {
        'dampening': 0, 'lr': 0.1, 'nesterov': False, 'weight_decay': 0, 'momentum': 0, 
        'params': [tensor([[0.6614, 0.2669], [0.0617, 0.6213]], requires_grad=True)]
    }
]

optimizer.param_groups is
[
    {
        'dampening': 0, 'lr': 0.1, 'nesterov': False, 'weight_decay': 0, 'momentum': 0, 
        'params': [tensor([[0.6614, 0.2669],[0.0617, 0.6213]], requires_grad=True)]
    }, 
    {
        'dampening': 0, 'lr': 0.0001, 'nesterov': False, 'weight_decay': 0, 'momentum': 0, 
        'params': [tensor([[-0.4519, -0.1661, -1.5228], [ 0.3817, -1.0276, -0.5631], [-0.8923, -0.0583, -0.1955]], requires_grad=True)]
    }
]
```
可以看到，上面两组参数拥有不同的学习率。


## .state_dict
```python
class Optimizer(object):
    def state_dict(self):
        ...
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }
```
**功能**：获取优化器当前状态信息**字典**


代码示例：
```python
# ----------------------------------- state_dict -----------------------------------
# flag = 0
flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print("state_dict before step:\n", opt_state_dict)

    for i in range(10):
        optimizer.step()

    print("state_dict after step:\n", optimizer.state_dict())

    torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))


# output:
state_dict before step:
{
    'param_groups': [{'params': [140186313140480], 'nesterov': False, 'weight_decay': 0, 'lr': 0.1, 'dampening': 0, 'momentum': 0.9}], 
    'state': {}
}

state_dict after step:
{
    'param_groups': [{'params': [140186313140480], 'nesterov': False, 'weight_decay': 0, 'lr': 0.1, 'dampening': 0, 'momentum': 0.9}], 
    'state': {140186313140480: {'momentum_buffer': tensor([[6.5132, 6.5132], [6.5132, 6.5132]])}}
}
```


## .load_state_dict
```python
class Optimizer(object):
    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}
```
**功能**：加载状态信息字典

&emsp; state_dict 和 load_state_dict 是一个组合操作，state_dict 用于获取优化器当前状态信息字典，而load_state_dict将状态字典加载到优化器中。
通常每隔几个epoch的就要保存一次模型的状态信息，避免训练意外终止后要从头训练的苦恼。


代码示例：
```python
# -----------------------------------load state_dict -----------------------------------
# flag = 0
flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    state_dict = torch.load(os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

    print("state_dict before load state:\n", optimizer.state_dict())
    optimizer.load_state_dict(state_dict)
    print("state_dict after load state:\n", optimizer.state_dict())


# output:
state_dict before load state:
{
    'param_groups': [{'params': [140186313140480], 'nesterov': False, 'weight_decay': 0, 'lr': 0.1, 'dampening': 0, 'momentum': 0.9}], 
    'state': {}
}

state_dict after load state:
{
    'param_groups': [{'params': [140186313140480], 'nesterov': False, 'weight_decay': 0, 'lr': 0.1, 'dampening': 0, 'momentum': 0.9}], 
    'state': {140186313140480: {'momentum_buffer': tensor([[6.5132, 6.5132], [6.5132, 6.5132]])}}
}
```


# 完整的训练代码示例
```python
# 参见：《DataLoader and Dataset》 所在章节。
```

下图是程序在运行完 `optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)` 后的内存状态：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/torch_Optimizer3.jpg" width = 100% height = 100% />
</div><center>图2 &nbsp;  优化器中的存储内容展示</center>


# 参考文献
[1] DeepShare.net > PyTorch框架
