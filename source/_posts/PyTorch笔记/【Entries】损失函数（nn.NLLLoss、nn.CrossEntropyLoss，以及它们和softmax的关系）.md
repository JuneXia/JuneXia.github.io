---
title: 【PyTorch笔记】损失函数（nn.NLLLoss、nn.CrossEntropyLoss，以及它们和softmax的关系）
date: 2019-11-02
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

话不多说，直接上代码。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 生成一个2x3的矩阵，假设这是模型预测值，表示有2条预测数据，每条是一个3维的激活值
inputs_tensor = torch.FloatTensor([
[10, 3,  1],
[-1, 0, -4]
])

# 真实值
targets_tensor = torch.LongTensor([1, 2])
```

有关softmax和交叉熵的理论知识可参见我的其他文章，或者参考网络。
```python
# 手动计算softmax
# ***********************************************************
inputs_exp = inputs_tensor.exp()
inputs_exp_sum = inputs_exp.sum(dim=1)
inputs_exp = inputs_exp.transpose(0, 1)
softmax_result = torch.div(inputs_exp, inputs_exp_sum)  # torch.div的两个输入张量必须广播一致的，而这两个张量的类型必须是一致的。
softmax_result = softmax_result.transpose(0, 1)
# ***********************************************************

>>>softmax_result
>>>tensor([[0.9990, 0.0009, 0.0001],
        [0.2654, 0.7214, 0.0132]])
```

## 
```python
# 使用F.softmax计算softmax
softmax_result = F.softmax(inputs_tensor)

>>>softmax_result
>>>tensor([[0.9990, 0.0009, 0.0001],
        [0.2654, 0.7214, 0.0132]])
```

```python
# 使用np.log计算得到log_softmax
log_softmax_result = np.log(softmax_result.data)
print('使用np.log计算得到log_softmax: ', softmax_result)

>>>log_softmax_result
>>>tensor([[-0.0010, -7.0010, -9.0010],
        [-1.3266, -0.3266, -4.3266]])
```

## F.log_softmax
```python
# 直接调用F.log_softmax计算得到log_softmax
log_softmax_result = F.log_softmax(inputs_tensor)

>>>log_softmax_result
>>>tensor([[-0.0010, -7.0010, -9.0010],
        [-1.3266, -0.3266, -4.3266]])
```

到这里我们可以看出，F.log_softmax的计算结果和先计算softmax再取log的效果是一样的。


```python
# 手动计算交叉熵损失
# ***********************************************************
_targets_tensor = targets_tensor.view(-1, 1)
onehot = torch.zeros(2, 3).scatter_(1, _targets_tensor, 1)  # 对真实标签做one-hot编码
product = onehot*log_softmax_result
cross_entropy = -product.sum(dim=1)
cross_entropy_loss = cross_entropy.mean()

>>>cross_entropy_loss
>>>tensor(5.6638)
# ***********************************************************
```

## nn.NLLLoss
```python
# 函数接口
class torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean') 
# weight: 权重列表，常用于解决类别不平衡问题；
```

NLLLoss全名是负对数似然损失函数（Negative Log Likelihood），在PyTorch的文档中有如下说明：

> Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.

简单来说，如果最后一层做了log softmax处理，那就可以直接使用nn.NLLLoss来计算交叉熵。
```python
# 使用nn.NLLLoss()计算log_softmax得到交叉熵损失
loss = nn.NLLLoss()
cross_entropy_loss = loss(log_softmax_result, targets_tensor)

>>>cross_entropy_loss
>>>tensor(5.6638)
```

## nn.CrossEntropyLoss
```python
# 直接使用nn.CrossEntropyLoss计算交叉熵损失
loss = nn.CrossEntropyLoss()
cross_entropy_loss = loss(inputs_tensor, targets_tensor)

>>>cross_entropy_loss
>>>tensor(5.6638)
```
致此，我们可以看出nn.CrossEntropyLoss()计算出的交叉熵和先计算softmax再使用F.NLLLoss()计算交叉熵的效果是一样的，所以从函数功能上看，nn.CrossEntropyLoss() = softmax+F.NLLLoss()
