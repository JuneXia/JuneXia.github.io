---
title: 
date: 2020-04-1
tags:
categories: ["深度学习笔记"]
mathjax: true
---
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) \
Mark Sandler Andrew Howard Menglong Zhu Andrey Zhmoginov Liang-Chieh Chen \
Google Inc. \
{sandler, howarda, menglong, azhmogin, lcchen}@google.com

代码已贴注释，论文整理未完待续。
<!-- more -->

**Abstract**
&emsp; In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. 
&emsp; is based on an **inverted residual structure** where the shortcut connections are between the thin bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. 
&emsp; Finally, our approach allows decoupling(decouple去耦合;使分离) of the input/output domains from the expressiveness(n. 善于表现;表情丰富;表现) of the transformation, which provides a convenient framework for further analysis. We measure our performance on ImageNet [1] classification, COCO object detection [2], VOC image segmentation [3]. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as actual latency, and the number of parameters.

# Introduction
&emsp; Neural networks have revolutionized many areas of machine intelligence, enabling superhuman accuracy for challenging image recognition tasks. However, the drive to improve accuracy often comes at a cost: modern state of the art networks require high computational resources beyond the capabilities of many mobile and embedded applications.

&emsp; This paper introduces a new neural network architecture that is specifically tailored(tailor v.专门制作,定制;(裁缝)度身缝制(衣服);使适应,迎合) for mobile and resource constrained environments. Our network pushes the state of the art for mobile tailored computer vision models, by significantly decreasing the number of operations and memory needed while retaining the same accuracy. 

&emsp; Our main contribution is a novel layer module: the **inverted residual with linear bottleneck**. 
This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. 
该模块采用低维压缩representation作为输入，该representation首先会被扩展到高维，然后用轻量级的depthwise convolution进行滤波。
Features are subsequently projected back to a low-dimensional representation with a linear convolution. The official implementation is available as part of TensorFlow-Slim model library in [4]. 

&emsp; This module can be efficiently implemented using standard operations in any modern framework and allows our models to beat(n.拍子;敲击;vt.打败;搅拌;adj.筋疲力尽的;疲惫不堪的) state of the art along multiple performance points using standard benchmarks. 
这个模块可以在任何现代框架中使用标准操作来有效地实现，并且使用标准benchmarks，可以让我们的模型在多个性能点上超越当前技术水平。
Furthermore, this convolutional module is particularly suitable for mobile designs, because it allows to significantly reduce the memory footprint needed during inference by `never fully materializing large intermediate tensors. (不完全实现大型中间张量)`. 
This reduces the need for `main memory(主存)` access in many embedded hardware designs, that provide small amounts of very fast software controlled cache memory.

# Related Work
&emsp; Tuning(n. 调音;音调;(电子或收音机)调谐;协调一致) deep neural architectures to strike(v.撞击;打;行进;达到(平衡)) an optimal balance between accuracy and performance `has been an area of active research(积极研究的一个领域)` for the last several years. 
调整深度神经结构以在精度和性能之间取得最佳平衡，是过去几年积极研究的一个领域。
Both manual architecture search and improvements in training algorithms, `carried out(实施,贯彻)` by numerous teams has lead to dramatic improvements over early designs such as AlexNet [5], VGGNet [6], GoogLeNet [7], and ResNet [8]. 
手工架构搜索以及大量团队对一些训练算法的改进，导致了对早期算法的巨大改进，
Recently there has been lots of progress in algorithmic architecture exploration included hyperparameter optimization [9, 10, 11] as well as various methods of network pruning(prune n.修剪;剪枝) [12, 13, 14, 15, 16, 17] and connectivity learning [18, 19]. \
A substantial(n. 本质;重要材料;adj.大量的;实质的) amount of work has also been dedicated(dedicate v. 致力;献身;题献;把…用于) to changing the connectivity structure of the internal convolutional blocks such as in ShuffleNet [20] or introducing sparsity [21] and others [22].
大量工作还致力于改变内部卷积块的connectivity结构，如ShuffleNet或引入稀疏性和其他。

&emsp; Recently, [23, 24, 25, 26], opened up a new direction of bringing optimization methods including genetic algorithms and reinforcement learning to architectural search. 
近年来，[23,24,25,26]开辟了一个新方向，包括将遗传算法、强化学习等优化方法引入到架构搜索。
However one drawback is that the resulting networks end up very complex. In this paper, we pursue the goal of developing better intuition(n.直觉;直觉力;直觉的知识) about how neural networks operate and use that to guide the simplest possible network design. Our approach should be seen as complimentary(adj.补充;赠送的;称赞的;问候的) to the one described in [23] and related work. 
我们的方法应该被看作是对[23]和相关工作中所描述的方法的补充。
`In this vein(在这方面)` our approach is similar to those taken by [20, 22] and allows to further improve the performance, while providing a glimpse(n.一瞥,一看; vt.瞥见) on its internal operation. 
在这方面，我们的方法类似于[20,22]所采取的方法，并允许进一步改进性能，同时提供对其内部操作的一瞥。
Our network design is based on MobileNetV1 [27]. It retains its simplicity and does not require any special operators while significantly improves its accuracy, achieving state of the art on multiple image classification and detection tasks for mobile applications.

未完待续。。。


PyTorch 官方实现代码解析

torchvison.models.mobilenet.py
```python
from torch import nn
from .utils import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    # 确保向下取整的时候不会超过10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    :param in_planes: the number of input channel.
    :param out_planes: the number of output channel.
    :param kernel_size:
    :param stride:
    :param groups: it's denote general converlution while groups=1, depth-wise converlution while groups=in_planes.
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        # hidden_dim is matched to $tk$ of paper.

        self.use_res_connect = self.stride == 1 and inp == oup
        # set using residul connect or not.

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            # In practicaly, layers 的 append or extend 是等效的，只不过append是一次追加一个元素，而extend是一次追加多个元素
            layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t: 扩展因子，用于在pw卷积中用于将k个输入channel数量变换到tk个输出channel。
                # c: 输出channel
                # n: InvertedResidual需要重复的次数
                # s: 卷基层stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # 在每个inverted residual blocks中，第一个卷积的stride用s，第二个卷积的stride都统一的用1

                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # avg-pooling
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

