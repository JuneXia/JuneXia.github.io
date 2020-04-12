---
title: 
date: 2020-03-31
tags:
categories: ["深度学习笔记"]
mathjax: true
---

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications \
Andrew G. Howard Menglong Zhu Bo Chen Dmitry Kalenichenko
Weijun Wang Tobias Weyand Marco Andreetto Hartwig Adam \
Google Inc.
{howarda,menglong,bochen,dkalenichenko,weijunw,weyand,anm,hadam}@google.com
<!-- more -->

**Abstract**
&emsp; We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses `depthwise separable convolutions` to build light weight deep neural networks. 
> depthwise separable convolutions, 深度可分离卷积
> wise：明智的;聪明的;博学的，

We introduce two simple global hyperparameters that efficiently `trade off(权衡)` between latency(n. 潜伏;潜在因素;延迟,时延) and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across(穿过;向) a wide range of applications and use cases including object detection, finegrain(n. 细晶粒,细致纹理) classification, face attributes and large scale geo-localization(地理定位).


# Introduction
&emsp; Convolutional neural networks have become ubiquitous in computer vision ever since AlexNet [19] popularized deep convolutional neural networks by winning the ImageNet Challenge: ILSVRC 2012 [24]. The general(n. 一般;将军;常规; adj. 一般的，普通的；综合的；大体的) trend(n.v. 趋势;倾向;走向) has been to make deeper and more complicated networks in order to achieve higher accuracy [27, 31, 29, 8]. However, these advances(advance n. 前进;预付款;求爱;v. 提出;使前进;提前 to improve accuracy are not necessarily making networks more efficient with respect to size and speed. In many real world applications such as robotics, self-driving car and augmented reality, the recognition tasks need to `be carried out(被执行;得到实现;进行;贯彻;开展)` in a timely(adj.及时的;适时的; adv. 及时地;早) fashion(n. 时尚;样式;方式; vt. 使用;改变) on a computationally limited platform.

&emsp; This paper describes an efficient network architecture and a set of two hyper-parameters `in order to(为了)` build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision applications. Section 2 reviews prior work in building small models. Section 3 describes the MobileNet architecture and two hyper-parameters width multiplier(n. [数]乘数;[电子]倍增器;增加者;倍频器) and resolution multiplier to define smaller and more efficient MobileNets. Section 4 describes experiments on ImageNet as well a variety of different applications and use cases. Section 5 closes with a summary and conclusion.

# Prior Work
&emsp; There has been rising interest in building small and efficient neural networks in the recent literature, e.g. [16, 34, 12, 36, 22]. 
Many different approaches can be generally categorized into either compressing pretrained networks or training small networks directly. 
许多不同的方法可以大致分为压缩预训练网络和直接训练小型网络。
`This paper(本文)` proposes a class of network architectures that allows a model developer to specifically choose a small network that matches the resource restrictions (latency, size) for their application. MobileNets primarily focus on optimizing for latency but also yield small networks. Many papers on small networks focus only on size but do not consider speed.

&emsp; MobileNets are built primarily from depthwise separable convolutions initially introduced in [26] and subsequently used in Inception models [13] to reduce the computation in the first few layers. 
Flattened networks [16] build a network out of fully factorized(factorize vt. 因式分解;把复杂计算分解为基本运算) convolutions and showed the potential of extremely factorized networks. 
扁平网络[16]构建了一个完全因数分解卷积的网络，并展示了高度因数分解网络的潜力。
Independent of this current paper, Factorized Networks[34] introduces a similar factorized convolution as well as the use of topological connections. Subsequently, the Xception network [3] demonstrated how to `scale up(向上扩展)` depthwise separable filters to out perform Inception V3 networks. Another small network is Squeezenet [12] which uses a bottleneck approach to design a very small network. Other reduced computation networks include structured transform networks [28] and deep fried convnets [37].
> fried &ensp; adj. 油炸的，油煎的；喝醉了的;  v. 油炸（fry的过去分词）

&emsp; `A different approach(另一种方法)` for obtaining small networks is shrinking(v. 缩水;收缩;缩小;退缩), factorizing or compressing pretrained networks. Compression based on `product quantization(乘积量化)[36]`, hashing(哈希算法)[2], and pruning(修剪;剪枝), `vector quantization(矢量量化)` and `Huffman coding(霍夫曼编码)`[5] have been proposed in the literature.
> quantization  n. [量子] 量子化;分层;数字化;量化
> hashing  散列法 散列 哈希算法

Additionally various factorizations have been proposed to speed up pretrained networks [14, 20]. Another method for training small networks is distillation(n. 蒸馏,净化;精华) [9] which uses a larger network to teach a smaller network. It is complementary(adj. 补足的;(基因序列等)互补的;辅助性的) to our approach and is covered in some of our use cases in section 4. 
它是对我们的方法的补充，并在第4节中介绍了我们的一些用例。
Another emerging(adj. 走向成熟的;新兴的; v. 浮现) approach is `low bit networks(低比特网络)` [4, 22, 11].

# MobileNet Architecture
&emsp; In this section we first describe the core layers that MobileNet is built on which are depthwise separable filters. We then describe the MobileNet network structure and conclude with descriptions of the two model shrinking hyperparameters width multiplier and resolution multiplier.

## Depthwise Separable Convolution
&emsp; The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a $1 \times 1$ convolution called a pointwise(逐点的) convolution. For MobileNets the depthwise convolution applies a single filter to each input channel. The pointwise  convolution then applies a $1 \times 1$ convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically(adv. 彻底地;激烈地;大幅度地) reducing computation and model size. Figure 2 shows how a standard convolution 2(a) is factorized into a  depthwise convolution 2(b) and a $1 \times 1$ pointwise convolution 2(c).

&emsp; A standard convolutional layer takes as input a $D_F \times D_F \times M$ feature map $\textbf{F}$ and produces a $D_F \times D_F \times N$ feature map $\textbf{G}$ where $D_F$ is the spatial width and height of a square input feature map, $M$ is the number of input channels (input depth), $D_G$ is the spatial width and height of a square output feature map and $N$ is the number of output channel (output depth).

&emsp; The standard convolutional layer is parameterized by convolution kernel $\textbf{K}$ of size $D_K \times D_K \times M \times N$ where $D_K$ is the spatial dimension of the kernel assumed to be square and $M$ is number of input channels and $N$ is the number of output channels as defined previously.

&emsp; The output feature map for standard convolution assuming stride one and padding is computed as:
$$
\mathbf{G}_{k,l,n} = \sum_{i,j,m} \mathbf{K}_{i,j,m,n} \cdot \mathbf{F}_{k+i-1, l+j-1, m}  \tag{1}
$$

&emsp; Standard convolutions have the computational cost of:
$$
D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F  \tag{2}
$$

where the computational cost depends multiplicatively(adv. 用乘法;积空间) on the number of input channels $M$, the number of output channels $N$ the kernel size $D_k \times D_k$ and the feature map size $D_F \times D_F$ . MobileNet models address each of these terms and their interactions(interaction n. [计] 交互,相互作用;相互交流). First it uses depthwise separable convolutions to break the interaction between the number of output channels and the size of the kernel.

&emsp; The standard convolution operation has the effect of filtering features based on the convolutional kernels and combining features in order to produce a new representation. The filtering and combination steps can be split into two steps via the use of factorized convolutions called depthwise separable convolutions for `substantial(n. 本质;重要材料; adj. 大量的;实质的;内容充实的) reduction in computational cost. (大幅降低计算成本)`.

&emsp; Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions. We use depthwise convolutions to apply a single filter per each input channel (input depth). Pointwise convolution, a simple $1 \times 1$ convolution, is then used to create a linear combination(n. 结合;组合;联合;[化学]化合) of the output of the depthwise layer. MobileNets use both batchnorm and ReLU nonlinearities for both layers. 

&emsp; Depthwise convolution with one filter per input channel (input depth) can be written as:
$$
\mathbf{\hat{G}}_{k,l,m} = \sum_{i,j} \mathbf{\hat{K}}_{i,j,m} \cdot \mathbf{F}_{k+i-1, l+j-1, m}  \tag{3}
$$

where $\mathbf{\hat{K}}$ is the depthwise convolutional kernel of size $D_K \times D_K \times M$ where the $m_{th}$ filter in $\mathbf{\hat{K}}$ is applied to the $m_{th}$ channel in $\mathbf{F}$ to produce the $m_{th}$ channel of the filtered output feature map $\mathbf{\hat{G}}$ .

&emsp; Depthwise convolution has a computational cost of:
$$
D_K \cdot D_K \cdot M \cdot D_F \cdot D_F  \tag{4}
$$
&emsp; Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via $1 \times 1$ convolution is needed in order to generate these new features.

&emsp; The combination of depthwise convolution and $1 \times 1$ (pointwise) convolution is called depthwise separable convolution which was originally introduced in [26].

&emsp; Depthwise separable convolutions cost:
$$
D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F  \tag{5}
$$
which is the sum of the depthwise and $1 × 1$ pointwise convolutions.

&emsp; By expressing convolution as a two step process of filtering and combining we get a reduction in computation of:
$$
\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F} {D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1} {N} + \frac{1} {D^2_K}
$$
&emsp; MobileNet uses $3 × 3$ depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy as seen in Section 4.

&emsp; Additional factorization in spatial dimension such as in [16, 31] does not save much additional computation as very little computation is spent in depthwise convolutions.
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/mobilenetv1-1.jpg" width = 60% height = 60% />
</div>


## Network Structure and Training
&emsp; The MobileNet structure is built on depthwise separable convolutions as mentioned(mention v. 提及,说起,谈到) in the previous section `except for(除…以外)` the first layer which is a full convolution. By defining the network in such simple terms we are able to easily explore network topologies to find a good network. The MobileNet architecture is defined in Table 1. All layers are followed by a batchnorm [13] and ReLU nonlinearity with the exception of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification. Figure 3 contrasts a layer with regular convolutions, batchnorm and ReLU nonlinearity to the factorized layer with depthwise convolution, $1 \times 1$ pointwise convolution as well as batchnorm and ReLU after each convolutional layer. Down sampling is handled with strided convolution in the depthwise convolutions as well as in the first layer. A final average pooling reduces the spatial resolution to 1 before the fully connected layer. Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers.
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/mobilenetv1-2.jpg" width = 60% height = 60% />
</div>

&emsp; It is not enough to simply(adv. 简单地;仅仅;简直) define networks `in terms of(依据;按照;在…方面)` a small number of Mult-Adds.
仅仅用少量的乘加操作来定义网络是不够的。
 It is also important to make sure these operations can be efficiently implementable. For instance `unstructured sparse matrix operations(非结构化稀疏矩阵操作)` are not typically faster than dense matrix operations until a very high level of sparsity. Our model structure puts nearly all of the computation into dense $1 \times 1$ convolutions. This can be implemented with highly optimized general matrix multiply (GEMM) functions. 
 Often convolutions are implemented by a GEMM but require an initial reordering(reorder n.v. 重新安排;重新排序;再订购) in memory called im2col in order to map it to a GEMM. 
通常，卷积是由GEMM实现的，但是需要在内存中进行名为im2col的初始重新排序才能将其映射到GEMM。
For instance, this approach is used in the popular Caffe package [15]. 
$1 \times 1$ convolutions do not require this reordering in memory and can be implemented directly with GEMM which is one of the most optimized numerical linear algebra(n. 代数,代数学) algorithms. 
……，GEMM是最优化的数值线性代数算法之一。\
MobileNet spends 95% of it's computation time in $1 \times 1$ convolutions which also has 75% of the parameters as can be seen in Table 2. Nearly all of the additional parameters are in the fully connected layer.

&emsp; MobileNet models were trained in TensorFlow [1] using RMSprop [33] with asynchronous(adj. [电]异步的;不同时的) gradient descent similar to Inception V3 [31]. 
However, contrary(n. 相反;反面; adj. 相反的;对立的;adv. 相反地) to training large models we use less regularization and data augmentation techniques because small models have less trouble(麻烦;烦恼;故障;动乱) with overfitting. 
然而，与训练大型模型相反，我们使用较少的正则化和数据扩充技术，因为小型模型的过拟合问题较少。\
When training MobileNets we do not use side heads or label smoothing and additionally reduce the amount image of distortions by limiting the size of small crops that are used in large Inception training [31]. 
当训练MobileNet的时候，我们不使用side heads或者标签平滑，并且通过限制small crops的大小来减少失真图片数量，这个small crops在大型 Inception 训练中会被用到。
Additionally, we found that it was important to put very little or no weight decay (l2 regularization) on the depthwise filters since their are so few parameters in them. For the ImageNet benchmarks(n. [计]基准;标竿;水准点; v. 测定基准点) in the next section all models were trained with same training parameters `regardless of(不顾,不管)` the size of the model.
> regardless &ensp; adj. 不管的,不顾的;不注意的；adv. 不顾后果地;不加理会;不管怎样,无论如何

## Width Multiplier: Thinner Models
&emsp; Although the base MobileNet architecture is already small and low latency, many times a specific use case or application may require the model to be smaller and faster. In order to construct these smaller and less computationally expensive models we introduce a very simple parameter $\alpha$ called width multiplier. The role of the width multiplier $\alpha$ is to thin a network uniformly at each layer. For a given layer and width multiplier $\alpha$, the number of input channels $M$ becomes $\alpha M$ and the number of output channels $N$ becomes $\alpha N$.

&emsp; The computational cost of a depthwise separable convolution with width multiplier $\alpha$ is:
$$
D_K \cdot D_K \cdot \alpha M \cdot D_F \cdot D_F + \alpha M \cdot \alpha N \cdot D_F \cdot D_F  \tag{6}
$$

where $\alpha \in (0, 1]$ with typical settings of 1, 0.75, 0.5 and 0.25. $\alpha = 1$ is the baseline MobileNet and $\alpha < 1$ are reduced MobileNets. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically(平方地;二次地;二次方) by roughly(adv. 粗糙地;概略地;大致) $\alpha^2$. Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, latency and size trade off. It is used to define a new reduced structure that needs to be trained from scratch.

## Resolution Multiplier: Reduced Representation
&emsp; The second hyper-parameter to reduce the computational cost of a neural network is a resolution multiplier $\rho$. We apply this to the input image and the internal representation of every layer is subsequently reduced by the same multiplier. In practice we implicitly set $\rho$ by setting the input resolution.

&emsp; We can now express the computational cost for the core layers of our network as depthwise separable convolutions with width multiplier $\alpha$ and resolution multiplier $\rho$:
$$
D_K \cdot D_K \cdot \alpha M \cdot \rho D_F \cdot \rho D_F + \alpha M \cdot \alpha N \cdot \rho D_F \cdot \rho D_F  \tag{7}
$$
where $\rho \in (0, 1]$ which is typically set implicitly so that the input resolution of the network is 224, 192, 160 or 128. $\rho = 1$ is the baseline MobileNet and $\rho < 1$ are reduced computation MobileNets. Resolution multiplier has the effect of reducing computational cost by $\rho^2$.

&emsp; As an example we can look at a typical layer in MobileNet and see how depthwise separable convolutions, width multiplier and resolution multiplier reduce the cost and parameters. Table 3 shows the computation and number of parameters for a layer as architecture shrinking methods are sequentially(adv. 从而;继续地;循序地) applied to the layer. The first row shows the Mult-Adds and parameters for a full convolutional layer with an input feature map of size $14 \times 14 \times 512$ with a kernel $K$ of size $3 \times 3 \times 512 \times 512$. We will look in detail in the next section at the trade offs between resources and accuracy.







