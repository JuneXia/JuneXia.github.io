---
title: 
date: 2020-04-16
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
<!-- more -->


# 父类 Sampler

首先需要知道的是所有的采样器都继承自Sampler这个类，如下：

可以看到主要有三种方法：分别是：
- **\_\_init\_\_**: 这个很好理解，就是初始化
- **\_\_iter\_\_**: 这个是用来产生迭代索引值的，也就是指定每个step需要读取哪些数据
- **\_\_len\_\_**: 这个是用来返回每次迭代器的长度

```python
class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)
```


# Sampler 的子类

## 顺序采样 SequentialSampler

这个看名字就很好理解，其实就是按顺序对数据集采样。

其原理是首先在初始化的时候拿到数据集data_source，之后在__iter__方法中首先得到一个和data_source一样长度的range可迭代器。每次只会返回一个索引值。

```python
class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
```

代码示例：
```python
import torch

max_epoches = 3

index_list = [1, 5, 78, 9, 68]
sampler = torch.utils.data.SequentialSampler(index_list)

for epoch in range(max_epoches):
    print('epoch', epoch, end=':\t')
    for idx in sampler:
        print(idx, end=' ')
    print('')

# output: 
epoch 0:	0 1 2 3 4 
epoch 1:	0 1 2 3 4 
epoch 2:	0 1 2 3 4 
```


## 随机采样 RandomSampler

参数说明：
- **data_source**: 同上
- **num_samples**: 指定采样的数量，默认是所有。
- **replacement**: 若为True，则表示可以重复采样，即同一个样本可以重复采样，这样可能导致有的样本采样不到。所以此时我们可以设置num_samples来增加采样数量使得每个样本都可能被采样到。

```python
class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples
```

代码示例：
```python
import torch

max_epoches = 3

index_list = [1, 5, 78, 9, 68]
sampler = torch.utils.data.RandomSampler(index_list, replacement=True, num_samples=7)


for epoch in range(max_epoches):
    print('epoch', epoch, end=':\t')
    for idx in sampler:
        print(idx, end=' ')
    print('')


# 改变 torch.utils.data.RandomSampler 的参数，会有以下输出结果：

# sampler = torch.utils.data.RandomSampler(index_list, replacement=False, num_samples=None)
epoch 0:	2 0 4 1 3 
epoch 1:	3 4 2 1 0 
epoch 2:	2 3 1 0 4

# sampler = torch.utils.data.RandomSampler(index_list, replacement=False, num_samples=None)
不允许的操作，会崩溃

# sampler = torch.utils.data.RandomSampler(index_list, replacement=True, num_samples=None)
epoch 0:	2 0 1 2 1 
epoch 1:	2 0 0 2 1 
epoch 2:	4 0 0 1 2 

# sampler = torch.utils.data.RandomSampler(index_list, replacement=True, num_samples=4)
epoch 0:	3 3 0 4 
epoch 1:	3 2 0 3 
epoch 2:	2 1 0 1

# sampler = torch.utils.data.RandomSampler(index_list, replacement=True, num_samples=7)
epoch 0:	4 1 1 1 0 3 4 
epoch 1:	1 2 0 1 4 0 0 
epoch 2:	1 4 0 2 2 4 3 
```


## 从子集采样 SubsetRandomSampler
```python
class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
```


这个采样器常见的使用场景是将训练集划分成训练集和验证集，示例如下：

```python
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.my_dataset import RMBDataset
import random
from libml.utils.config import SysConfig

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

flag = True
# flag = False
if flag:
    max_epoches = 3
    batch_size = 10
    split_dir = os.path.join(SysConfig['home_path'], 'res', 'RMB_data/rmb_split')
    train_dir = os.path.join(split_dir, "train")
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
    n_train = len(train_data)
    split = n_train // 3
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    # 注意：DataLoader如果使用sampler的话，是不能设置shuffle为True的
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    for epoch in range(max_epoches):
        print('epoch', epoch, end=':\t')
        for data in train_loader:
            print(data)
        print('')
```


## 带权重的随机采样 WeightedRandomSampler

参数说明：
- **weights**: 为每张图片赋予一个权重，且权重和数值大小无关，只与权重相互之间的比值有关
- **num_samples**: 同前
- **replacement**: 同前

```python
class WeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
```


使用示例：
```python
flag = True
# flag = False
if flag:
    max_epoches = 3
    batch_size = 10
    split_dir = os.path.join(SysConfig['home_path'], 'res', 'RMB_data/rmb_split')
    train_dir = os.path.join(split_dir, "train")
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)

    # ============================ 为每张图片赋予一个权重 ============================
    # 这里的权重与实际大小无关，只与相互之间的比值有关
    weights = []
    for _, label in train_data:
        if label == 0:
            weights.append(1)
        else:
            weights.append(2)

    n_train = len(train_data)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, n_train + 2, replacement=True)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    actual_labels = []
    for epoch in range(max_epoches):
        print('epoch', epoch)
        for i, data in enumerate(train_loader):
            print(i, data[1])
            actual_labels.extend(data[1].numpy().tolist())
        print('')

    print('0/1: {}'.format(actual_labels.count(0) / actual_labels.count(1)))


# output: 
epoch 0
0 tensor([1, 1, 0, 0, 1, 0, 0, 1, 1, 1])
...
15 tensor([1, 0, 0, 0, 0, 1, 1, 0, 1, 1])
16 tensor([1, 1])

epoch 1
...

epoch 2
...

# 3个epoch所取出的样本中标签为0和1的样本比例为0.5，符合我们设定的权重比值
0/1: 0.5046439628482973
```


## 批量采样器 BatchSampler

上面讲的采样器每次都只能采样一个样本(当然sampler通过DataLoader的封装后可以实现批量采样)，下面这个采样器可以每次产生一个批量的样本。

- **sampler**: 一个单输出采样器，可以用上面的采样器实例，BatchSampler通过这个sampler产生一个batch的样本索引
- **batch_size**: 使用BatchSampler需要设置batch_size，注意这时候在DataLoader中就不需要设置batch_size了
- **drop_last**: 

```python
class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```


使用示例：
```python
flag = True
# flag = False
if flag:
    max_epoches = 3
    batch_size = 10
    split_dir = os.path.join(SysConfig['home_path'], 'res', 'RMB_data/rmb_split')
    train_dir = os.path.join(split_dir, "train")
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)

    # ============================ 为每张图片赋予一个权重 ============================
    # 这里的权重与实际大小无关，只与相互之间的比值有关
    weights = []
    for _, label in train_data:
        if label == 0:
            weights.append(1)
        else:
            weights.append(2)

    n_train = len(train_data)

    index_list = [1, 5, 78, 9, 68]
    # sampler = torch.utils.data.SequentialSampler(list(range(len(train_data))))
    sampler = torch.utils.data.RandomSampler(list(range(len(train_data))), replacement=False, num_samples=None)
    train_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=3, drop_last=False)

    # 如果使用batch_sampler，则DataLoader的batch_size需要设置为1，shuffle需设置为False
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, batch_sampler=train_sampler)

    # 直接迭代train_sampler采样器，得到的是一个个索引值
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for data in train_sampler:
        print(data)
    print('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    actual_labels = []
    for epoch in range(max_epoches):
        print('epoch', epoch)
        for i, data in enumerate(train_loader):
            print(i, data[1])
        print('')

    print('0/1: {}'.format(actual_labels.count(0) / actual_labels.count(1)))
```

