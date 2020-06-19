---
title: 
date: 2020-6-15
tags:
categories: ["深度学习笔记"]
mathjax: true
---

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
Jun-Yan Zhu ∗, Taesung Park ∗, Phillip Isola, Alexei A. Efros \
Berkeley AI Research (BAIR) laboratory, UC Berkeley

2017年

**Abstract**
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence(n. 没有；缺乏；缺席；不注意) of paired examples. Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable(adj. 不能区别的，不能辨别的；不易察觉的) from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained(adj.约束过少的;未限定的), we couple(n. 对；夫妇；数个;vi. 结合；成婚;vt. 结合；连接；连合) it with an inverse mapping F : Y → X and introduce a cycle consistency(n. [计] 一致性；稠度；相容性) loss to enforce F (G(X)) ≈ X (and `vice versa(反之亦然)`). Qualitative(adj. 定性的；质的，性质上的) results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration(n. 变形；变容；变貌), season transfer, photo enhancement, etc. Quantitative(adj. 定量的；量的，数量的) comparisons against several prior methods demonstrate the superiority(n. 优越，优势；优越性) of our approach.

![](../../images/ml/CycleGAN-1.jpg)


# Introduction
&emsp; What did Claude Monet see as he placed his easel by the bank of the Seine near Argenteuil on a lovely spring day in 1873 (Figure 1, top-left)? A color photograph, had it been invented, may have documented a crisp blue sky and a glassy river reflecting it. Monet conveyed his impression of this same scene through wispy brush strokes and a bright palette.

&emsp; What if Monet had happened upon the little harbor in Cassis on a cool summer evening (Figure 1, bottom-left)? A brief stroll through a gallery of Monet paintings makes it possible to imagine how he would have rendered the scene: perhaps in pastel shades, with abrupt dabs of paint, and a somewhat flattened dynamic range.

&emsp; We can imagine all this despite never having seen a side by side example of a Monet painting next to a photo of the scene he painted. Instead, we have knowledge of the set of Monet paintings and of the set of landscape photographs. We can reason about the stylistic differences between these two sets, and thereby imagine what a scene might look like if we were to “translate” it from one set into the other.

![](../../images/ml/CycleGAN-2.jpg)

&emsp; In this paper, we present a method that can learn to do the same: capturing special characteristics of one image collection and figuring out how these characteristics could be translated into the other image collection, all in the absence of any paired training examples.

&emsp; This problem can be more broadly described as image-to-image translation [22], converting an image from one representation of a given scene, $x$, to another, $y$, e.g., grayscale to color, image to semantic labels, edge-map to photograph. Years of research in computer vision, image processing, computational photography, and graphics have produced powerful translation systems in the supervised setting, where example image pairs ${x_i, y_i}^N_{i=1}$ are available (Figure 2, left), e.g., [11, 19, 22, 23, 28, 33, 45, 56, 58, 62]. **However, obtaining paired training data can be difficult and expensive.** For example, only a couple of datasets exist for tasks like semantic segmentation (e.g., [4]), and they are relatively small. Obtaining input-output pairs for graphics tasks like artistic stylization can be even more difficult since the desired output is highly complex, typically requiring artistic authoring. For many tasks, like object transfiguration (e.g., zebra↔horse, Figure 1 top-middle), the desired output is not even well-defined.

&emsp; We therefore seek an algorithm that can learn to translate between domains without paired input-output examples (Figure 2, right). `We assume there is some underlying relationship between the domains – for example, that they are two different renderings(render n/v. 翻译；表现；表演；描写) of the same underlying scene – and seek to learn that relationship.(我们假设在域之间有一些潜在的关系(例如，它们是同一基本场景的两种不同的渲染), 并试图了解这种关系.` Although we lack supervision in the form of paired examples, we can exploit supervision at the level of sets: we are given one set of images in domain X and a different set in domain Y . We may train a mapping $G : X → Y$ such that the output $ŷ = G(x), x ∈ X$, is indistinguishable(adj. 不能区别的，不能辨别的；不易察觉的) from images $y ∈ Y$ by an adversary trained to classify $ŷ$ apart(adj/adv. 分离的/地；与众不同的/地) from $y$. \
In theory, this objective can induce an output distribution over $ŷ$ that matches the empirical distribution $p_{data}(y)$ (in general, this requires $G$ to be stochastic) [16]. \
这个目标能够推导出一个关于 $\hat{y}$ 的分布，从经验上来说该分布可以匹配 $p_{data}(y)$ 分布)   \
The optimal G thereby translates the domain X to a domain Ŷ distributed identically(adv. 同一地；相等地) to Y . \
优化G从而转换X域到Ŷ域，并且使Ŷ域的分布和Y域相同。\
`However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over ŷ. (然而这样的转换并不能保证单个的输入x和输出y是以一种有意义的方式匹配的，因为有无穷多个这种映射G能够实现这种操作).` \
`Moreover, in practice, we have found it difficult to optimize the adversarial objective in isolation: standard procedures often lead to the well-known problem of mode collapse, where all input images map to the same output image and the optimization fails to make progress [15].(此外，在实践中，我们发现很难单独对敌对目标进行优化:标准程序经常会导致众所周知的模式崩溃问题，即所有输入图像都映射到同一输出图像，优化无法取得进展).`

> Summary：以前的通过训练一个X域到Y域的映射 $G : X → Y$ 这种方法，并不能保证 X 和 Y 是有意义的。言外之意就是说本文后面将要提出的方法能够解决这一问题。

These issues call for(要求；需要；提倡；邀请；为…叫喊) adding more structure to our objective. Therefore, we exploit the property that translation should be “cycle consistent(adj. 始终如一的，一致的；坚持的)”, in the sense that if we translate, e.g., a sentence from English to French, and then translate it back from French to English, we should arrive back at the original sentence [3]. Mathematically, if we have a translator $G : X → Y$ and another translator $F : Y → X$, then $G$ and $F$ should be inverses(n/adj. 相反；倒转；反面) of each other, and both mappings should be bijections(n. [数]双射). We apply this structural assumption by training both the mapping G and F simultaneously, and adding a *cycle consistency(n. [计]一致性；稠度；相容性) loss* [64] that encourages $F(G(x)) ≈ x$ and $G(F(y)) ≈ y$. `Combining this loss with adversarial losses on domains X and Y yields our full objective for unpaired image-to-image translation.(将这种损失与X和Y域上的对抗损失结合起来，就得到了未配对图像到图像转换的完整目标.)` 
> Summary: 在非配对 image-to-image 转换任务中，要遵循 cycle consistent 原则，即通过G生成的图片要尽可能被F解回去，即$F(G(x)) ≈ x$ and $G(F(y)) ≈ y$

&emsp; We apply our method to a wide range of applications, including collection style transfer, object transfiguration(n. 变形；变容；变貌), season transfer and photo enhancement. `We also compare against previous approaches that rely either on hand-defined factorizations(n. [数]因式分解) of style and content, or on shared embedding functions,(我们还将以前的方法与依赖于手工定义的style和content的因式分解或共享的嵌入函数进行了比较),` and show that our method outperforms these  baselines. We provide both [PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [Torch](https://github.com/junyanz/CycleGAN) implementations. Check out more results at our [website](https://junyanz.github.io/CycleGAN/).
> summary: 将我们的方法应用在不同的生成式任上，结果表明我们的方法更好。


# Related work
&emsp; **Generative Adversarial Networks (GANs)** [16, 63] have achieved impressive results in image generation [6, 39], image editing [66], and representation learning [39, 43, 37]. Recent methods adopt the same idea for conditional image generation applications, such as text2image [41], image inpainting(图像修复;图像修补) [38], and future prediction [36], as well as to other domains like videos [54] and 3D data [57]. `The key to GANs’ success is the idea of an adversarial loss that forces the generated images to be, in principle, indistinguishable(adj.不能辨别的;不易察觉的) from real photos.(GANs成功的关键是对抗性缺失的理念，迫使生成的图像原则上与真实的照片无法区分).` `This loss is particularly(adv. 异乎寻常地；特别是；明确地) powerful for image generation tasks, as this is exactly(adv. 恰好地；正是；精确地；正确地) the objective that much of computer graphics aims to optimize.(这种损失对于图像生成任务尤其强大，因为这正是许多计算机图形优化的目标).` We adopt an adversarial loss to learn the mapping such that the translated images cannot be distinguished from images in the target domain.
> 近年来GAN在很多方面得到应用，GAN能够起作用的关键就是其“对抗损失”的训练思想。（具体什么对抗损失，本段也没细说）


&emsp; **Image-to-Image Translation** The idea of image-to-image translation goes back at least to Hertzmann et al.’s Image Analogies [19], who employ a non-parametric texture model [10] on a single input-output training image pair. More recent approaches use a dataset of input-output examples to learn a parametric translation function using CNNs (e.g., [33]). Our approach builds on the “pix2pix” framework of Isola et al. [22], which uses a conditional generative adversarial network [16] to learn a mapping from input to output images. Similar ideas have been applied to various tasks such as generating photographs from sketches(n.[测]草图;示意图;草图法(sketch的复数);v.素描,写生) [44] or from attribute and semantic layouts [25]. However, `unlike the above prior work(与上面的工作不同)`, we learn the mapping without paired training examples. 
> 简单介绍了下 image-to-image 方法的起源以及more recent进展，我们的方法是基于 pix2pix [22] 的，它使用GAN学习一个从输入到输出的映射，但是我们没有配对的训练样本。


**Unpaired Image-to-Image Translation** Several other methods also tackle(v. 应付,处理) the unpaired setting, where **the goal is to relate two data domains: X and Y**. <font face="黑体" color=red size=2>Rosales et al. [42] propose a **Bayesian framework** that includes a prior based on a patch-based **Markov random field** computed from a source image and a likelihood term obtained from multiple style images.</font> More recently, **CoGAN** [32] and cross-modal scene networks [1] use a weight-sharing strategy to learn a common representation across domains. Concurrent(adj. 并发的；一致的；同时发生的；并存的) to our method, Liu et al. [31] extends the above framework with a **combination of variational(adj. 变化的；因变化而产生的；[生物]变异的) autoencoders [27] and generative adversarial networks** [16]. Another line of concurrent work [46, 49, 2] **encourages the input and output to share specific “content” features even though they may differ in “style“**. These methods also use adversarial networks, with additional terms to enforce the output to be close to the input in a predefined metric space, such as class label space [2], image pixel space [46], and image feature space [49].
> summary: 介绍了几个其他处理非配对数据的方法，他们的目标都是关联X和Y两个域的数据，这些方法也使用对抗网络，只是额外的会强化输出和预定义的输入空间之间的联系。


&emsp; Unlike the above approaches, our formulation does not rely on any task-specific, predefined similarity function be-tween the input and output, nor do we assume that the input and output have to lie in the same low-dimensional embedding space. This makes our method a general-purpose solution for many vision and graphics tasks. We directly compare against several prior and contemporary approaches in Section 5.1.