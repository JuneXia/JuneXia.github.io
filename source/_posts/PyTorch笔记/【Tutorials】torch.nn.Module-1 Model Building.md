---
title: 
date: 2019-09-22
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
&emsp; 在《【Tutorials】autograd-2 Logistic Regression》一节中已经提到了机器学习模型训练的一般步骤，如下图所指示，关于数据部分我们在前面已经说过了，本节主要讲述模型构建相关的问题。
<!-- more -->

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/autograd_ml_train_step.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp;机器学习模型训练步骤</center>

&emsp; PyTorch的模型构建接口来自torch.nn，torch.nn主要有如下几个模块：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module2.jpg" width = 60% height = 60% />
</div>
<center>图2 &nbsp;torch.nn下的常用子模块</center>

本节主要讲述nn.Module模块。


# 使用nn.Module构建模型
&emsp; 模型的构建一般主要有以下几个组成部分：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module1.jpg" width = 60% height = 60% />
</div>
<center>图3 &nbsp;模型构建的组成部分</center>

每个nn.Module**实例**都有下面这8个字典来管理它的属性：
```python
self._parameters = OrderedDict()
self._buffers = OrderedDict()
self._backward_hooks = OrderedDict()
self._forward_hooks = OrderedDict()
self._forward_pre_hooks = OrderedDict()
self._state_dict_hooks = OrderedDict()
self._load_state_dict_pre_hooks = OrderedDict()
self._modules = OrderedDict()
```
> **注意**: OrderedDict()是一个有序的字典。

以上8个字典，其实总结下来就是下面的4类字典：\
- **parameters**: 存储管理nn.Parameter类的属性，而nn.Parameter又是继承自torch.Tensor，所以nn.Parameter的对象是一个个Tensor，例如weight、bias这些参数；
- **modules**: 存储管理nn.Module类的属性，例如卷积层、池化层等；
- **buffers**: 存储管理缓冲属性, 如 BN 层中的running_mean
- *****_hooks**: 存储管理钩子函数（一共有5个hook函数）


**nn.Module总结** \
- 一个module可以包含多个子module，例如LeNet可以包含卷积层、池化层这些子module等；
- 一个module相当于一个运算，必须实现forward()函数；
- 每个module都有8个字典管理它的属性

代码示例：参见之前的。


# 使用容器构建网络模型

## 使用nn.Sequential构建模型
nn.Sequential 是 nn.Moudle 的容器，用于**按顺序**包装一组网络层。
- **顺序性**：各网络层之间严格按照顺序构建
- **自带forward()**: 自带的forward里，通过for循环依次执行前向传播运算

### 使用默认命名构建模型
```python
# -*- coding: utf-8 -*-
"""
# @file name  : module_containers.py
# @author     : tingsongyu
# @date       : 2019-09-20 10:08:00
# @brief      : 模型容器——Sequential, ModuleList, ModuleDict
"""
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


# ============================ Sequential
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

```

&emsp; 上面LeNet的实现方法默认会使用整型值作为各个layer的key来构建整个network的，network会通过这些key来索引各个layer。但是在更大型的网络中这种构建方法会很难通过整型key来索引 layer，所以有时候需要使用下面的有序字典 OrderedDict 来构建network的各个layer.

### 使用有序字典构建模型
```python
class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

# net = LeNetSequential(classes=2)
# net = LeNetSequentialOrderDict(classes=2)
#
# fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
#
# output = net(fake_img)
#
# print(net)
# print(output)
```

## 使用nn.ModuleList构建module
nn.ModuleList是nn.module的容器，用于包装一组网络层，以**迭代**方式调用网络层
- **append()**: 在ModuleList后面**添加**网络层
- **extend()**: **拼接**两个ModuleList
- **insert()**: 指定在ModuleList中位置**插入**网络层

```python
# ============================ ModuleList

class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


# net = ModuleList()
#
# print(net)
#
# fake_data = torch.ones((10, 10))
#
# output = net(fake_data)
#
# print(output)
```

## 三种容器构建方式总结
nn.Sequential: **顺序性**，各网络层之间严格按照顺序执行，常用于block的构建（子模块），当一个子模块是固定的时候就可以用nn.Sequential来构建。
nn.ModuleList: **迭代性**，常用大量重复网络的构建，通过for循环实现重复构建
nn.ModuleDict: **索引性**，常用于可选择的网络层

```python
# ============================ ModuleDict

class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


net = ModuleDict()

fake_img = torch.randn((4, 10, 32, 32))

output = net(fake_img, 'conv', 'relu')

print(output)
```

# AlexNet构建

AlexNet: 2012年以高出第二名10多个百分点的准确率获得ImageNet分类任务冠军,开创了卷积神经网络的新时代。
AlexNet特点如下:
1. 采用ReLU: 替换饱和激活函数, 减轻梯度消失
2. 采用LRN(Local Response Normalization): 对数据归一化, 减轻梯度消失（后来被BN所取代）
3. Dropout: 提高全连接层的鲁棒性, 增加网络的泛化能力
4. Data Augmentation: TenCrop、色彩修改

参考文献:《ImageNet Classification with Deep Convolutional Neural Networks》

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/nn_Module4_AlexNet.jpg" width = 60% height = 60% />
</div>
<center>图4 &nbsp;AlexNet网络结构</center>


# 参考文献
[1] DeepShare.net > PyTorch框架
