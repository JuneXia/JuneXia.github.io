---
title: 
date: 2019-09-20
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要介绍PyTorch中的数据标准化(Normalize)方法。
<!-- more -->

# 为什么要对数据进行标准化？
&emsp; 因为对数据标准化后可以加快模型的收敛。我们这里借助《【Tutorials】autograd-2 Logistic Regression》中的代码，通过改变 bias 查看对训练结果的影响。

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/autograd_logistic_regression2.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/autograd_logistic_regression3.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图1 &nbsp;逻辑回归 bias=1 的训练结果</center>
</html>

<br>

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_normalize1.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_normalize2.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图2 &nbsp;逻辑回归 bias=5 的训练结果</center>
</html>

&emsp; 模型参数的初始化值一般是在均值为0方差为1的正态分布附近，，图1中的数据离这个分布更近，而图2中的数据离这个分布更远，所以就算使用图2中数据来训练，模型最后也能够收敛，但这个收敛速度相对使用图1中的数据就要慢很多了，而且图2中的模型最后收敛的效果也没有图1好。（图1迭代380次acc便可达到99.5%，Loss等于0.0493；而图2迭代到980次时acc才达到99.0%，Loss等于0.1469. 图1完胜图2）


# PyTorch中的数据标准化：transforms.Normalize
&emsp; PyTorch中的数据标准化方法是通过视觉工具包torchvision中的transforms.Normalize方法实现的。

```python
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
```

**功能**：逐channel的对图像进行标准化，计算公式如下：$output = (input - mean) / std$

- **mean**: 各通道的均值
- **std**: 各通道的标准差
- **inplace**: 是否原地操作


# Code Examples
&emsp; 完整代码可参见《【Tutorials】DataLoader and Dataset》.

```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),  # 将图片转为张量，并做归一化操作(归一化到0~1区间)
    transforms.Normalize(norm_mean, norm_std),
])
```


# 参考文献
[1] DeepShare.net > PyTorch框架
