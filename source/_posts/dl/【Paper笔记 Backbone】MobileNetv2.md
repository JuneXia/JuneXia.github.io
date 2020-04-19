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
&emsp; is based on an **inverted residual structure**(倒残差结构) where the shortcut connections are between the thin bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. 
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

# Preliminaries, discussion and intuition
## Depthwise Separable Convolutions
&emsp; Depthwise Separable Convolutions are a key building block for many efficient neural network architectures [27, 28, 20] and we use them in the present work as well. The basic idea is to replace a full convolutional operator with a factorized(factorize vt.因式分解) version that splits convolution into two separate layers. The first layer is called a **depthwise convolution**, it performs lightweight filtering by applying a single convolutional filter per input channel. The second layer is a $1 \times 1$ convolution, called a **pointwise convolution**, which is responsible for building new features through computing linear combinations of the input channels.
&emsp; Standard convolution takes an $h_i × w_i × d_i$ input tensor $L_i$, and applies convolutional kernel $K \in R^{k×k×d_i×d_j}$ to produce an $h_i × w_i × d_j$ output tensor $L_j$. Standard convolutional layers have the computational cost of $h_i · w_i · d_i · d_j · k · k$.

&emsp; Depthwise separable convolutions are a drop-in replacement for standard convolutional layers. Empirically they work almost as well as regular convolutions but only cost:
$$
h_i \cdot w_i \cdot d_i (k^2 + d_j)  \tag{1}
$$
这个式子的由来（我的理解）：
$h_i \cdot w_i \cdot d_i \cdot k \cdot k + h_i \cdot w_i \cdot d_i \cdot 1 \cdot 1 \cdot d_j$，前面是depthwise卷积的计算量，后面是pointwise卷积的计算量。

&emsp; which is the sum of the depthwise and $1 × 1$ pointwise convolutions. Effectively depthwise separable convolution reduces computation compared to traditional layers by almost a factor of $k^2$ (more precisely(adv.精确地;恰恰), by a factor $k^2d_j/(k^2 + d_j)$). 

这个结论的简单推理如下：
$$
\begin{aligned}
    \frac{h_i \cdot w_i \cdot d_i (k^2 + d_j)} {h_i · w_i · d_i · d_j · k · k} = \frac{k^2 \cdot d_j} {k^2 + d_j}
\end{aligned}
$$
假设卷积核 $k$ 设置为3，$d_j = 100$，则上式等于：$\frac{9 \cdot 100} {9 + 100} \approx \frac{900}{100} = 9 = 3^2$ \

MobileNetV2 uses $k = 3$ ($3 \times 3$ depthwise separable convolutions) so the computational cost is 8 to 9 times smaller than that of standard convolutions at only a small reduction in accuracy [27].

## Linear Bottlenecks
&emsp; Consider a deep neural network consisting of $n$ layers $L_i$ each of which has an activation tensor of dimensions $h_i \times w_i \times d_i$. Throughout this section we will be discussing the basic properties of these activation tensors, which we will treat as containers of $h_i \times w_i$ pixels with $d_i$ dimensions. Informally(adv. 非正式地;不拘礼节地;通俗地), for an input set of real images, we say that the set of layer activations (for any layer $L_i$) forms a “manifold of interest”. 
> manifold \
> adv. 非常多
> n. （汽车引擎用以进气和排气）歧管，多支管；有多种形式之物；流形
> adj. 多种多样的，许多种类的
> v. 复印；使……多样化

It has been long assumed that manifolds of interest in neural networks could be embedded in low-dimensional subspaces. 
长期以来，人们一直认为对神经网络感兴趣的流形可以嵌入到低维子空间中。
In other words, when we look at all individual d-channel pixels of a deep convolutional layer, `the information encoded in those values actually lie in some manifold(这些值中编码的信息实际上是一些流形)`, which in turn is embeddable into a low-dimensional subspace.
> Note that dimensionality of the manifold differs from the dimensionality of a subspace that could be embedded via a linear transformation.

&emsp; At a first glance(v.瞥闪,瞥见,扫视,匆匆一看;浏览), such a fact could then be captured and exploited(exploit vt. 开发,开拓;剥削;开采;n.勋绩;功绩) by simply reducing the dimensionality of a layer thus reducing the dimensionality of the operating space. 
乍一看，这样一个事实可以通过简单地减少一个层的维度来捕获和利用，从而减少操作空间的维度。

`This has been successfully exploited by MobileNetV1`[27] to effectively trade off between computation and accuracy via a width multiplier parameter, and has been incorporated into efficient model designs of other networks as well [20]. 
`MobileNetV1 已经成功地利用这一点`，通过宽度乘子参数有效地在computation和accuracy之间进行权衡，并且这一方法已被纳入到其他网络以及[20]的高效模型设计中。

Following that intuition(n.直觉), the width multiplier approach allows one to reduce the dimensionality of the activation space until the manifold of interest spans this entire(adj.全部的;全体的) space. 
根据这种直觉，宽度乘子方法允许我们降低激活空间的维度，直到interest manifold跨越整个空间。

However, this intuition breaks down when we recall that deep convolutional neural networks actually have non-linear per coordinate transformations, such as ReLU. 
然而，当我们回想起深度卷积神经网络实际上具有每个坐标的非线性转换(如ReLU)时，这种直觉就失效了。

For example, ReLU applied to a line in 1D space produces a 'ray', where as in $R^n$ space, it generally results in a `piece-wise(adj.[数]分段的;adv.分段地)` linear curve with $n$-joints.
例如，ReLU应用到一维空间中的一条直线上会产生一条射线，而在$R^n$空间中，它通常会产生具有$n$个关节的分段线性曲线。

&emsp; It is easy to see that in general if a result of a layer transformation ReLU($Bx$) has a non-zero volume $S$, the points mapped to interior $S$ are obtained via a linear transformation $B$ of the input, thus indicating that the part of the input space corresponding to the full dimensional output, is limited to a linear transformation. 
通常很容易看到，一个layer变换ReLU($Bx$)的结果是有一个非 0 volume $S$，那么映射到内部$S$的点是通过输入的一个线性变换$B$得到的，从而表明，与full dimensional输出相对应的输入空间部分，被限制为一个线性变换。

In other words, deep networks only have the power of a linear classifier on the non-zero volume part of the output domain. We refer to supplemental material for a more formal statement.
换句话说，深度网络只在输出域的非零volume部分具有线性分类器的能力。

👆？

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/mobilenetv2-1.jpg" width = 80% height = 80% />
</div>
Figure 1: &emsp; Examples of ReLU transformations of low-dimensional manifolds embedded in higher-dimensional spaces. In these examples the initial spiral(n.螺旋;旋涡;adj.螺旋形的;盘旋的) is embedded into an $n$-dimensional space using random matrix $T$ followed by ReLU, and then projected back to the 2D space using $T^{-1}$. In examples above n = 2,3 result in information loss where certain points of the manifold collapse into each other, while for n = 15 to 30 the transformation is highly `non-convex(非凸)`.

> 上面这段话的意思实际就是说：作者做了一些实验案例，即使用随机矩阵 $T$ 和 ReLU 将初始初始的2维螺旋形嵌入到一个n维空间中，然后使用 $T^{-1}$ 将其投影回2D空间。而当n=2或3时，即螺旋形先被嵌入到2或3维空间中然后再被投影回2D空间，这会导致某些点塌陷(变形)，而当n=15到30时，这种塌陷(变形)会有所缓解。
> 
> 而这实际上就是说，对低维特征使用ReLU这种非线性激活函数会导致严重的信息损失。
> 
> 从而道出了作者在MobileNetv2中使用线性激活函数替代非线性激活函数的原因：
> 由于倒残差结构的输入、输出都是低维特征，而用ReLU这种非线性激活函数会导致信息损失比较严重，所以作者在MobileNetv2中用线性激活函数替代了非线性激活函数。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/mobilenetv2-2.jpg" width = 80% height = 80% />
</div>
Figure 2: Evolution(n.演变;进化论;进展) of separable convolution blocks. The diagonally(adv.对角地;斜对地) hatched(hatch n.v.孵化;策划; hatched adj.阴影线的) texture indicates layers that do not contain non-linearities. The last (lightly colored) layer indicates the beginning of the next block. Note: 2d and 2c are equivalent(相等的;等价的) blocks when stacked(stack n.v.堆叠). Best viewed in color.

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/mobilenetv2-3.jpg" width = 80% height = 80% />
</div>
Figure 3: The difference between residual block [8, 30] and inverted residual. Diagonally hatched layers do not use non-linearities. We use thickness(n.厚度;层;浓度;含混不清) of each block to indicate its relative(n.相关物;adj.相对的) number of channels. Note how classical(adj.经典的;传统的) residuals connects the layers with high number of channels, whereas the inverted residuals connect the bottlenecks. Best viewed in color.

&emsp; On the other hand, when ReLU collapses the channel, it inevitably loses information in *that channel*. However if we have lots of channels, and there is a a structure in the activation manifold that information might still be preserved in the other channels. In supplemental(adj.补充的(等于supplementary);追加的) materials, we show that if the input manifold can be embedded into a significantly(adv.显著地;相当数量地) lower-dimensional subspace of the activation space then the ReLU transformation preserves the information while introducing the needed complexity into the set of expressible functions.

&emsp; To summarize, we have highlighted two properties that are indicative(adj.象征的;指示的;表示…的) of the requirement that the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space:
综上所述，我们强调了两个性质，这两个性质表明了我们所关心的流形应该位于高维激活空间的低维子空间中：

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of preserving complete information about the input manifold, but only if the input  manifold lies in a low-dimensional subspace of the input space. \
ReLU能够保存输入流形的完整信息，但前提是输入流形位于输入空间的低维子空间中。

&emsp; These two insights provide us with an `empirical hint(经验提示)` for optimizing existing neural architectures: assuming the manifold of interest is low-dimensional we can capture this by inserting *linear bottleneck* layers into the convolutional blocks. Experimental evidence suggests that using linear layers is crucial as it prevents(prevent v.防止;阻止;预防) nonlinearities from destroying(destroy vt.破坏;消灭;毁坏) too much information. In Section 6, we show empirically that using non-linear layers in bottlenecks indeed hurts the performance by several percent, `further validating our hypothesis.(从而进一步验证了我们的假设)`. 
> We note that in the presence of shortcuts the information loss is actually less strong. \
> 我们注意到，在存在shortcuts的情况下，信息损失实际上不那么严重。

We note that similar reports where non-linearity was helped were reported in [29] where non-linearity was removed from the input of the traditional residual block and that lead to improved performance on CIFAR dataset.
> 蹩脚直译：\
> 我们注意到，在[29]中也有关于非线性层是有帮助的这样类似的报道，而non-linearity被从传统残差块中的输入中移除，从而提高了CIFAR数据集的性能。
> 
> 理解翻译：\
> 我们注意到，也有类似的报道，在[29]中报告说非线性是有帮助的，然而却将传统残差块输入中的非线性移除，从而提高了CIFAR数据集的性能。(言外之意就是说：非线性确实有帮助，但在[29]中还是将其移除了，而移除了效果会更好)

&emsp; For the remainder(n.[数]余数,残余;剩余物;其余的人; adj.剩余的; v.廉价出售) of this paper we will be utilizing bottleneck convolutions. 
We will refer to the ratio between the size of the input bottleneck and the inner size as the *expansion ratio*.
我们将输入瓶颈的大小与内部大小的比值称为扩展比。

## Inverted residuals
&emsp; The bottleneck blocks appear similar to residual block where each block contains an input followed by several bottlenecks then followed by expansion [8]. However, inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation(n.[计]实现;履行;实施) detail that accompanies a non-linear transformation of the tensor, we use shortcuts directly between the bottlenecks.

**Running time and parameter count for bottleneck convolution** &emsp; The basic implementation structure is illustrated in Table 1. For a block of size $h \times w$, expansion factor $t$ and kernel size $k$ with $d'$ input channels and $d''$ output channels, the total number of multiply add required is $h · w · d' · t(d' + k^2 + d'')$. Compared with (1)(指公式(1)) this expression has an extra term, as indeed we have an extra $1 \times 1$ convolution, however the nature of our networks allows us to utilize much smaller input and output dimensions. In Table 3 we compare the needed sizes for each resolution between MobileNetV1, MobileNetV2 and ShuffleNet.

## Information flow interpretation
&emsp; One interesting property of our architecture is that it provides a natural separation between the input/output domains of the building blocks (bottleneck layers), and the layer transformation that is a non-linear function that converts input to the output. The former(前者) can be seen as the capacity of the network at each layer, whereas the latter(后者) as the expressiveness(n. 表达能力;善于表现;表情丰富). 

This is in contrast with traditional convolutional blocks, both regular and separable, where both expressiveness and capacity are tangled(tangle v.(使)缠结在一起;(使)乱成一团;争吵;打架;n.缠结;混乱,纷乱;争吵;打架) together and are functions of the output layer depth.
这与传统的卷积块形成对比，无论是规则块还是可分块，在传统卷积块中，表达性和容量都是纠缠在一起的，是输出层深度的函数。

&emsp; In particular, in our case, when inner layer depth is 0 the underlying convolution is the identity function thanks to the shortcut connection. 
特别地，在我们的例子中，当内层深度为 0 时，由于快捷连接，底层卷积是 identity function. \
When the expansion ratio is smaller than 1, this is a classical residual convolutional block [8, 30]. However, for our purposes we show that expansion ratio greater than 1 is the most useful.

&emsp; This interpretation allows us to study the expressiveness of the network separately from its capacity and we believe that further exploration of this separation is warranted(warrant n.根据;证明;正当理由;委任状;vt.保证;担保;批准;辩解) to provide a better understanding of the network properties.
这一解释使我们能够独立于网络的容量来研究网络的表达性，我们相信，对这种分离的进一步探索有助于更好地理解网络的特性。


# Model Architecture


<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/mobilenetv2-4.jpg" width = 80% height = 80% />
</div>





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

