---
title: 
date: 2020-6-20
tags:
categories: ["深度学习笔记"]
mathjax: true
---
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) \
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros \
Berkeley AI Research (BAIR) Laboratory, UC Berkeley \
{isola,junyanz,tinghuiz,efros}@eecs.berkeley.edu

CVPR2017

**Abstract**
&emsp; We investigate conditional adversarial networks as a `general-purpose(通用的)` solution to image-to-image translation problems. These networks **not only** learn the mapping from input image to output image, **but also learn a loss function to train this mapping**. This makes it possible to apply the same generic(adj.一般的,通用的;属的;非商标的) approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking(tweak v.扭,捏,拧;稍稍改进,对…稍作调整). `As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable(adj.合理的,公道的;通情达理的) results without hand-engineering our loss functions either. (作为一个社区，我们不再手工设计我们的映射函数，而这项工作表明，我们也可以无需手工设计我们的损失函数就可以实现理想的结果).`
> summary: 我们使用Conditional-GAN来作为image-to-image的解决方案，我们的网络不仅会学习input到output的映射，也会学习训练这个映射的损失函数。

# Introduction
&emsp; Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input image into a corresponding output image. `Just as a concept may be expressed in either English or French (正如一个概念既可以用英语也可以用法语表达一样)`, a scene may be rendered as an RGB image, a gradient field, an edge map, a semantic label map, etc. In analogy to automatic language translation, we define automatic image-to-image translation as the task of translating one possible representation of a scene into another, given sufficient training data (see Figure 1). Traditionally, each of these tasks has been tackled(tackle 解决,处理,对付) with separate, special-purpose machinery (e.g., [16, 25, 20, 9, 11, 53, 33, 39, 18, 58, 62]), despite the fact that the setting is always the same: predict pixels from pixels. Our goal in this paper is to develop a common framework for all these problems.
> summary: 就像language translation(语言翻译)一样，我们这里定义image-to-image 的转换问题。基本都是废话。


&emsp; The community has already taken significant steps in this direction, with convolutional neural nets (CNNs) becoming the common workhorse(n. 做重活的人;adj. 工作重的；吃苦耐劳的) behind a wide variety of image prediction problems. `CNNs learn to minimize a loss function – an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still goes into designing effective losses. (尽管学习过程是自动的,但cnn学会了将损失函数最小化,这是一个衡量结果质量的目标, 设计有效损失仍然需要大量的手工工作.)` In other words, we still have to tell the CNN what we wish it to minimize. But, just like King Midas, we must be careful what we wish for! If we take a naive(adj.天真的,幼稚的) approach and ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry(adj.模糊的，失去焦距的) results [43, 62]. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring. \
`Coming up with (提出；想出；赶上)` loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images – is an open problem and generally requires expert knowledge. \
提出损失功能，迫使CNN做我们真正想要的，例如，输出锐利的、逼真的图像是一个开放的问题，通常需要专家的知识.
> summary: CNN 现在已经成了各种图像预测问题的常用工具，但是有效的损失函数仍然需要大量的手工操作。（主要就是要表达损失函数的设计很重要）


&emsp; `It would be highly desirable if we could instead specify only a high-level goal, like “make the output indistinguishable(adj. 不能辨别的；不易察觉的) from reality”, and then automatically learn a loss function appropriate for satisfying this goal. (如果我们能指定一个高层次的目标，比如将输出与现实区分开来，然后自动学习一种适合于满足这个目标的损失功能，那将是非常可取的.)` Fortunately, this isexactly what is done by the recently proposed Generative Adversarial Networks (GANs) [24, 13, 44, 52, 63]. GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss.  Blurry images will not be tolerated since they look obviously fake. `Because GANs learn a loss that adapts to the data, they can be applied to a multitude(n. 大量，多数；群众，人群) of tasks that traditionally would require very different kinds of loss functions. (由于GAN会学习适应数据的损失，因此它们可以应用于许多任务上，而这些任务如果要用传统方法去做 通常需要非常不同种类的损失函数.)`
> summary: GAN的损失很棒，GAN 会学习一个能够辨别输出图片真假的损失函数（其实就是判别器），同时训练一个生成器来减小这个loss (生成器)。GAN 的这些损失函数在[24, 13, 44, 52, 63]中已有所提及。
> 
> **值得注意的是：**\
> 其实整个判别器可以看成是一个损失函数，只不过这个损失函数比较复杂，它是一个网络。


&emsp; In this paper, we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model[24]. This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image.
> summary: GANs 学习一个生成模型，cGANs 学习一个条件生成模型。这使得cGANs适合于做 image-to-image 任务，而我们可以有条件的控制输入图片从而产生相应的输出图片。

&emsp; GANs have been vigorously(adj.精力充沛的,有力的) studied in the last two years and many of the techniques we explore in this paper have been previously proposed. Nonetheless(adv.尽管如此,但是), earlier papers have focused on specific applications, and it has remained unclear how effective image-conditional GANs can be as a `general-purpose (通用的)` solution for image-to-image translation. **Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results. Our second contribution is to present a simple framework sufficient(adj.足够的,充分的) to achieve good results, and to analyze the effects of several important architectural choices.** Code is available at https://github.com/phillipi/pix2pix.
> summary: 早期的GAN更多的是关注特定的应用，但是尚不清楚以image为conditional的 GANs 在 image-to-image 上的效果如何。我们的贡献如上文加粗字体所示。


# Related work
&emsp; **Structured losses for image modeling** Image-to-image translation problems are often formulated as per-pixel classification or regression (e.g.,[39, 58, 28, 35, 62]). These formulations treat the output space as “unstructured” in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a structured loss. Structured losses penalize(vt. 处罚；处刑；使不利) the joint configuration(n. 配置;结构;布局;外形) of the output. A large body of literature has considered losses of this kind, with methods including conditional random fields [10], the SSIM metric [56], feature matching [15], nonparametric losses [37], the convolutional pseudo-prior [57], and losses based on matching covariance(n.[数]协方差；共分散) statistics [30]. The conditional GAN is different in that **the loss is learned**, and can, in theory, penalize any possible structure that differs between output and target.
> summary: image-to-image 问题经常被定义为逐像素的分类或回归问题，这种论述会把输出空间当作非结构化来处理，在这种情况下对于给定的输入图片，每个输出像素之间被认为是相互独立的。而conditional-GANs学习一个结构化的损失，结构化损失能够惩罚输出的联合外形结构。已经有大量的文献考虑到了这种损失，但是conditional-GANs的不同在于它的loss是被学习到的，并且从理论上讲，conditional-GANs可以惩罚输出和目标之间任何不同的可能结构。

&emsp; **Conditional GANs** We are not the first to apply GANs in the conditional setting.  Prior and concurrent(同时代) works have conditioned GANs on discrete(adj. 离散的，不连续的; n. 分立元件；独立部件) labels [41, 23, 13], text [46], and, indeed, images. The image-conditional models have tackled(tackle 解决，处理，对付) image prediction from a normal map [55], future frame prediction [40], product photo generation [59], and image generation from sparse(adj.稀少的, 稀疏的) annotations(n.注解, 评注) [31, 48] (c.f. [47]for an autoregressive approach to the same problem). Several other papers have also used GANs for image-to-image mappings, but only applied the GAN unconditionally, relying on other terms (such as L2 regression) to force the output to be conditioned on the input.  These papers have achieved impressive results on inpainting [43], future state prediction [64], image manipulation(n. 操纵；操作；处理；篡改) guided by user constraints(n. 约束；限制；约束条件) [65], style transfer [38], and superresolution [36]. Each of the methods was tailored(tailor v.专门制作,定制;使适应,迎合) for a specific application. `Our framework differs in that nothing is application-specific. This makes our setup considerably simpler than most others. (我们的框架的不同之处在于，没有什么是特定于应用程序的。这使我们的设置比大多数其他设置简单得多。)`
> summary: 我们不是第一个在GANs 中使用条件设置的，但是之前的方法都是针对某个特定的应用的。而我们的不同之处就是在于我们的方法并不是针对某个特定应用。


&emsp; Our method also differs from the prior works in several architectural choices for the generator and discriminator. Unlike past work, for our generator we use a “U-Net”-based architecture [50], and for our discriminator we use a convolutional “PatchGAN” classifier, which only penalizes structure at the scale of image patches. A similar PatchGAN architecture was previously proposed in [38] to capture local style statistics. Here we show that this approach is effective on a wider range of problems, and we investigate the effect of changing the patch size.
> summary: 在 generator 和 discriminator 的网络结构选择上，我们也有一些不同，generator 基于U-Net，discriminator 基于 PatchGAN，我们还研究了不同pacth size的影响。


[](../../images/ml/pix2pix-2.jpg)
Figure 2: Training a conditional GAN to map edges photo. The discriminator, D, learns to classify between fake (synthesized by the generator) and real {edge, photo} tuples. The generator, G, learns to fool the discriminator. **Unlike an unconditional GAN, both the generator and discriminator observe the input edge map.**
> summary: 和非条件GAN不同，条件GAN的generator和discriminator都要观察输入的边缘图。


# Method
&emsp; GANs are generative models that learn a mapping from **random noise vector z to output image y**, G : z → y [24]. In contrast, conditional GANs learn a mapping from **observed image x and random noise vector z, to y,** G : {x,z} → y. The generator G is trained to produce outputs that cannot be distinguished from “real” images by an adversarially trained discriminator, D, which is trained to do as well as possible at detecting the generator’s “fakes”. This training procedure is diagrammed in Figure 2.
> summary: 对比了GAN和条件GAN的不同，GAN是 G : z → y，其中z是vector，y是image；而条件GAN是 G : {x, z} → y，其中z是vector，x和y都是image。


## Objective
&emsp; The objective of a conditional GAN can be expressed as
$$
\mathcal{L}_{cGAN} (G, D) = \mathbb{E}_{x,y} [ \text{log} D(x, y)] + \mathbb{E}_{x,z} [\text{log} (1 -D (x, G(x, z)))],  \tag{1}
$$
where $G$ tries to minimize this objective against an adversarial $D$ that tries to maximize it, i.e. $G^∗ = \text{arg min}_G \text{max}_D \ \mathcal{L}_{cGAN} (G, D)$.

To test the importance of conditioning the discriminator, we also compare to an unconditional variant in which the discriminator does not observe $x$:
$$
\mathcal{L}_{GAN} (G, D) = \mathbb{E}_{y} [ \text{log} D(y)] + \mathbb{E}_{x,z} [\text{log} (1 - D(G(x, z)))].  \tag{2}
$$

Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance [43]. The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:
$$
\mathcal{L}_{L1} (G) = \mathbb{E}_{x,y,z} [ \Vert y - G(x, z) \Vert_1 ].  \tag{3}
$$
> summary: 之前的方法发现混合使用 GAN 目标损失和一个更传统的损失(如L2距离)会更有益。现在我们让鉴别器的job保持不变，但生成器的任务是不仅仅要欺骗鉴别器，还要在L2损失下更接近Ground-Truth。我们还探索了使用L1来代替L2，因为L1鼓励产生更少的模糊。


Our final objective is
$$
G^* = \text{arg} \min \limits_{G} \max \limits_{D} \mathcal{L}_{cGAN} (G, D) + \lambda \mathcal{L}_{L1} (G).  \tag{4}
$$

&emsp; Without z, the net could still learn a mapping from x to y, but would produce deterministic(adj.确定性的;命运注定论的) outputs, and therefore fail to match any distribution other than a delta function.
没有z，网络仍然可以学习从x到y的映射，但会产生听天由命的输出，因此无法匹配除delta函数以外的任何分布。\
Past conditional GANs have acknowledged this and provided Gaussian noise z as an input to the generator, `in addition to(除了…之外(还有，也))` x (e.g., [55]). In initial(最初的) experiments, we did not find this strategy effective – the generator simply(adv.简单地;仅仅;简直) learned to ignore the noise – which is consistent(adj.始终如一的,一致的;坚持的) with Mathieu et al. [40]. Instead, **for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time.**
> summary: 没有z，网络仍然可以学习从x到y的映射，但是所产生的y却是听天由命的了。过去的条件GANs也意识到了这一点，所以除了x之外还提供了高斯噪声z作为generator的输入。但在最初的实验中，我们并没有发现这个策略有效，因为generator似乎会学着忽略这一噪声，这一点跟Mathieu等人的想法[40]也一致。**而对于我们最终的模型，我们只使用了dropout形式的噪声。**

`Despite the dropout noise, we observe only minor(adj.未成年的;次要的;较小的) stochasticity(随机性) in the output of our nets.(尽管有dropout噪声，但是在我们的网络输出中，我们观察到只有较小的随机性.)` \
`Designing conditional GANs that produce highly stochastic output, and thereby(adv. 从而,因此) capture the full entropy of the conditional distributions they model, is an important question left(adj. 左边的;左派的;剩下的) open by the present work. (设计能够产生高度随机输出的条件GANs，从而获得它们所建模的条件分布的完全熵，是目前工作遗留下来的一个重要问题.)`


## Network architectures
&emsp; We adapt our generator and discriminator architectures from those in [44]. Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu [29]. Details of the architecture are provided in the supplemental(adj.补充的;附加的) materials online, with key features discussed below.

### Generator with skips
A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution output grid. `In addition, for the problems we consider, the input and output differ in surface appearance, but both are renderings of the same underlying structure. (另外，对于这个问题我们还会考虑：输入和输出拥有不同的表面外观，但都有相同的底层结构的渲染)` Therefore, structure in the input is roughly(adv.粗糙地;概略地) aligned with structure in the output. We design the generator architecture around these considerations.
> summary: 如上重要。

&emsp; Many previous solutions [43, 55, 30, 64, 59] to problems in this area have used an encoder-decoder network [26]. In such a network, the input is passed through a series of layers that progressively(adv.渐进地;日益增多地) downsample, until a bottleneck layer, at which point the process is reversed(reverse v.颠倒;翻转). Such a network requires that all information flow pass through all the layers, including the bottleneck. For many image translation problems, there is a great deal of low-level information shared between the input and output, and `it would be desirable(adj.可取的,值得拥有的,令人向往的) to shuttle(n. 航天飞机;穿梭;梭子;公共汽车等) this information directly across the net (因此直接通过网络传输这些信息是可取的).` For example, in the case of image colorization(着彩色;灰度图着色;颜色迁移), `the input and output share the location of prominent(adj. 突出的，显著的；杰出的；卓越的) edges. (输入和输出共享突出边缘的位置).`
> summary: 在这一领域，之前的一些解决方案是使用一个 encoder=decoder 网络，在这样的网络中，输入经过一系列渐近下采样，直到一个bottleneck层。对于许多image translation问题，输入和输出之间有大量的低级共享信息。
> 
> **？？？。。。**


&emsp; To give the generator a means to circumvent(v. 包围；智取；绕行，规避) the bottleneck for information like this, we **add skip connections, following the general shape of a “U-Net”** [50]. Specifically, we add skip connections between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i.
> summary: 为了给 generator 提供绕过此类信息瓶颈的方法，我们按照U-Net[50]的一般形状增加跳跃连接。具体来说，我们在每一层i和第n-i层之间增加跳跃连接，其中n是层的总数。每个跳跃连接只是将第i层的所有通道与第n-i层的通道连接起来。


### Markovian discriminator (PatchGAN)
It is well known that the L2 loss – and L1, see Figure 4 – produces blurry results on image generation problems [34]. Although these losses fail to encourage high-frequency crispness(n.易碎;清新;酥脆), in many cases they nonetheless(adv.尽管如此,但是) accurately capture the low frequencies. `For problems where this is the case (对于这样的问题)`, we do not need an entirely new framework to enforce correctness at the low frequencies. L1 will already do.
> summary: 众所周知，尽管L1和L2损失在图像生成问题上会产生图像模糊，但是我们完全不需要新的框架，因为L1已经够用了。

&emsp; This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term(n.术语;学期;期限;条款;(代数式等的)项;vt.把…叫做) to force low-frequency correctness (Eqn. 4). \
如果只是依赖L1项来保证低频的正确性(Eqn.4)，那么这就限制了GAN鉴别器只对高频结构建模。\
In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. \
为了对高频进行建模，只关注局部图片patch中的结构就足够了。\
Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes `structure at the scale of patches (patch尺度的结构)`. This discriminator tries to classify if each N ×N patch in an image is real or fake. `We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate(n. 终极；根本；基本原则; adj. 最终的；极限的；根本的) output of D. (我们在图像上卷积运行这个鉴别器，平均所有的响应，以提供最终的D输出.)`
> summary: L1损失只能保证低频信息的正确性（也就是说L1只能生成轮廓大体相似的图片，但生成的图片整体比较模糊，没有细节特征），而对高频信息的建模效果较差。所以我们这里设计了一种只关注局部信息的鉴别器——我们称之为PatchGAN。

&emsp; In Section 4.4, we demonstrate that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily(adv.任意地;武断地;反复无常地;专横地) large images. 
> summary: 如上。

&emsp; Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter(n.直径). This connection was previously explored in [38], and is also the common assumption in models of texture [17, 21] and style [16, 25, 22, 37]. Therefore, our PatchGAN can be understood as a form of texture/style loss.
> summary: 假设各个patch之间是相互独立的，那么我们上面说的PatchGAN这样一个鉴别器就能够有效地将图片建模为一个马尔可夫随机场。这种联系之前在[38]种已经有所探索，并且这也是纹理模型[17,21]和风格模型[16,25,22,37]种常见的假设。因此我们的PatchGAN也可以被理解为一种纹理/风格形式的损失。
> 
> **这似乎有个缺点(me)：**
> 为了保证生成的图片中的目标在整体上显得比较自然一些，各个patch之间不可能相互独立啊，否则生成的图片会有段截的感觉。


## Optimization and inference
&emsp; To optimize our networks, we follow the standard approach from [24]: we alternate between one gradient descent step on D, then one step on G. `As suggested in the original GAN paper, rather than training G to minimize log(1 − D(x, G(x, z)), we instead train to maximize log D(x, G(x, z)) [24]. (正如在original GAN 论文中所建议的，我们不是训练G最小化log(1 − D(x, G(x, z))，而是训练将log D(x, G(x, z))最大化 [24]).` In addition, `we divide the objective by 2 while optimizing D, which slows down the rate at which D learns relative to G. (我们在优化D时将目标除以2，从而降低D对相对于G的学习速率).` We use minibatch SGD and apply the Adam solver [32], with a learning rate of 0.0002, and momentum parameters β1 = 0.5, β2 = 0.999.
> summary: 如上。


&emsp; At inference time, we run the generator net in exactly(adv.恰好地;正是;精确地;正确地) the same manner(n. 方式；习惯；种类；规矩；风俗) as during the training phase. This differs from the usual protocol in that **we apply dropout at test time**, and `we apply batch normalization [29] using the statistics of the test batch, rather than aggregated(aggregate v. 集合;聚集;合计) statistics of the training batch. (我们应用BN使用的是test batch的统计信息，而不是train batch的统计信息集合).` This approach to batch normalization, when the batch size is set to 1, has been termed “instance normalization” and has been demonstrated to be effective at image generation tasks [54]. In our experiments, we use batch sizes between 1 and 10 depending on the experiment.
> summary: 重要如上。


# Experiments


## Evaluation metrics


## Analysis of the objective function


## Analysis of the generator architecture


## From PixelGANs to PatchGANs to ImageGANs


## Perceptual validation


## Semantic segmentation


## Community-driven Research


# Conclusion




