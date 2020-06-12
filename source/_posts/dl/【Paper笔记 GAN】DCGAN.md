---
title: 
date: 2020-6-12
tags:
categories: ["深度学习笔记"]
mathjax: true
---

[UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)

Alec Radford & Luke Metz
indico Research
Boston, MA
{alec,luke}@indico.io

Soumith Chintala
Facebook AI Research
New York, NY
soumith@fb.com

2016年

<!--more-->

**ABSTRACT**
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. **We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints(constraint n. [数]约束;限制;约束条件), and demonstrate that they are a strong candidate for unsupervised learning.** Training on various image datasets, we show convincing(adj.令人信服的;有说服力的; convince v. 使相信;使明白) evidence that our deep convolutional adversarial pair learns a hierarchy(n. 层级;等级制度) of representations from object parts to scenes in both the generator and discriminator. \
通过对各种图像数据集的训练，我们有令人信服的证据表明：我们的深度卷积对抗pair在Generator和Discriminator中学习到了从目标部分到场景的表示层次。\
Additionally, we use the learned features for novel tasks - demonstrating their applicability(n. 适用性;适应性) as general image representations.
此外，我们将学习到的特征用于新任务，以证明它们在一般图像再现语句中的适用性。


# INTRODUCTION
Learning reusable(adj.可重复使用的) feature representations from large unlabeled datasets has been an area of active research. In the context of computer vision, `one can leverage(n. 手段，影响力；杠杆作用；v. 利用；举债经营) the practically(adv. 实际地；几乎；事实上) unlimited amount of unlabeled images and videos(人们几乎可以利用无限量未标记的图片和视频 )` to learn good intermediate representations, which can then be used on a variety of supervised learning tasks such as image classification. We propose that one way to build good image representations is by training Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks. \
`GANs provide an attractive(adj.吸引人的;有魅力的) alternative to maximum likelihood techniques. (GANs为最大似然技术提供了一个有吸引力的替代方案.)`. \
One can `additionally argue(另外说)` that their learning process and the lack of a heuristic(adj.启发式的;探索的) cost function (such as pixel-wise independent mean-square error) are attractive to representation learning. \
此外，他们的学习过程和缺乏启发式代价函数(如像素独立均方误差)对表示学习很有吸引力。\
GANs have been known to be unstable to train, often resulting in generators that produce nonsensical(adj. 无意义的；荒谬的) outputs. There has been very limited published research in trying to understand and visualize what GANs learn, and the intermediate representations of multi-layer GANs.

In this paper, we make the following contributions
- We propose and evaluate a set of constraints on the architectural topology of Convolutional GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANs (DCGAN)
- We use the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms.
- We visualize the filters learnt by GANs and empirically(adv.以经验为主地) show that specific filters have learned to draw specific objects.
- `We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples. (我们展示了生成器具有有趣的向量算术属性，允许对生成的样本的许多语义质量进行简单的操作.)`



# RELATED WORK

## representationl earning from unlabeled data (从未标记的数据中获益)
Unsupervised representation learning is a fairly(adv. 相当地；公平地；简直) well studied problem in general computer vision research, as well as in the context of images. **A classic approach to unsupervised representation learning is to do clustering on the data** (for example using K-means), and leverage(n. 手段，影响力；杠杆作用；v. 利用；举债经营) the clusters for improved classification scores. In the context of images, one can do hierarchical clustering of image patches (Coates & Ng, 2012) to learn powerful image representations. **Another popular method is to train auto-encoders** (convolutionally, stacked (Vincent et al., 2010), separating the what and where components of the code (Zhao et al., 2015), ladder(n. 阶梯；途径；梯状物；vi. 成名；发迹) structures (Rasmus et al., 2015)) that encode an image into a compact code, and decode the code to reconstruct the image as accurately as possible. These methods have also been shown to learn good feature representations from image pixels. **Deep belief networks** (Lee et al., 2009) have also been shown to work well in learning hierarchical representations.


## generatting natural images

Generative image models are well studied and `fall into(落入；分成)` two categories: parametric and non-parametric. 

**The non-parametric models often do matching from a database of existing images, often matching patches of images**, and have been used in **texture synthesis** (Efros et al., 1999), **super-resolution** (Freeman et al., 2002) and `in-painting(图像修复)` (Hays & Efros, 2007). 
> paint v. 绘画；涂色于 \
> painting n. 绘画；油画；着色 \

**Parametric models for generating images has been explored(explore v.开发;探寻;冒险) extensively(adv. 广阔地；广大地) (for example on MNIST digits or for texture synthesis** (Portilla & Simoncelli, 2000)). However, generating natural images of the real world have had not much success until recently. \
A variational(adj.变化的;因变化而产生的;[生物]变异的) sampling approach to generating images (Kingma & Welling, 2013) has had some success, but the samples often suffer from being blurry(adj. 模糊的；污脏的；不清楚的). \
Another approach generates images using an iterative(adj.[数]迭代的;重复的,反复的) forward(adv.向前地;按顺序地) diffusion(n.扩散,传播;[光]漫射) process (Sohl-Dickstein et al., 2015). \
Generative Adversarial Networks (Goodfellow et al., 2014) generated images suffering from being noisy and incomprehensible(adj.费解的;无限的;不可思议的). \
A laplacian(n.[数]拉普拉斯算子) pyramid extension(n.拓展;延伸) to this approach (Denton et al., 2015) showed higher quality images, but they still suffered from the objects looking wobbly(adj.不稳定的;歪斜的) `because of noise introduced in chaining multiple models (因为在多模型链接中引入了噪声.)`. \
A recurrent network approach (Gregor et al., 2015) and a deconvolution network approach (Dosovitskiy et al., 2014) have also recently had some success with generating natural images. However, they have not leveraged the generators for supervised tasks.


## visualizing the internals of cnns (cnn内部构件可视化)

One constant criticism of using neural networks has been that they are black-box methods, with little understanding of what the networks do in the form of a simple `human-consumable(人类可读的,可供人类消耗的)` algorithm. \
In the context of CNNs, Zeiler et. al. (Zeiler & Fergus, 2014) showed that by **using deconvolutions and filtering the maximal activations**, one can find the approximate purpose of each convolution filter in the network. \
Similarly, **using a gradient descent on the inputs lets us inspect(v.检查;视察) the ideal image that activates certain subsets of filters** (Mordvintsev et al.).
> 上面这段话就是说：人们常常批判CNN的不可解释性，所以会有人尝试去做一些CNN的可解释性研究.


# APPROACH AND MODEL ARCHITECTURE
Historical attempts to scale up GANs using CNNs to model images have been unsuccessful. This motivated the authors of LAPGAN (Denton et al., 2015) to develop an alternative approach to iteratively upscale low resolution generated images which can be modeled more reliably(adv.可靠地;确实地). \
有史以来，使用CNN做图像建模来 scale up GANs 的尝试一直很成功，这激励了LAPGAN的作者开发了一个可替换的方法，用于将低分辨率的图像放大，这种建模方式确实可靠。\
We also encountered difficulties attempting to scale GANs using CNN architectures commonly(adv.一般地;通常地) used in the supervised literature(n.文学;文献). However, after extensive model exploration we identified a family of architectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models.

Core to our approach is adopting and modifying three recently demonstrated changes to CNN architectures.

**The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic(adj.确定性的;命运注定论的) spatial pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn its own spatial downsampling.** \
第一，使用带strided的卷积网络来替换所有的空间池化函数(例如 maxpooling)，这允许网络能够学习它自己的空间下采样。 \
We use this approach in our generator, allowing it to learn its own spatial upsampling, and discriminator. \
我们在我们的Generator和Discriminator中都使用这种方法，其中Generator将被允许学习它自己的空间上采样过程，而Discriminator将被允许学习它自己的空间下采样过程。


**Second is the trend(n/v.趋势,倾向;走向) towards eliminating fully connected layers on top of convolutional features.** The strongest example of this is global average pooling which has been utilized in state of the art image classification models (Mordvintsev et al.). We found **global average pooling increased model stability but hurt convergence speed**. <font face="黑体" color=red size=2>A middle ground of directly connecting the highest convolutional features to the input and output respectively of the generator and discriminator worked well.</font> The first layer of the GAN, which takes a uniform noise distribution Z as input, **could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack**. **For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.** See Fig. 1 for a visualization of an example model architecture.

**Third is Batch Normalization** (Ioffe & Szegedy, 2015) which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models. This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs. **Directly applying batchnorm to all layers however, resulted in sample oscillation(n.振荡;振动;摆动) and model instability. This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.**

**The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function.** We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution. **Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well**, especially for higher resolution modeling. This is in contrast to the original GAN paper, which used the maxout activation (Goodfellow et al., 2013).

Architecture guidelines for stable Deep Convolutional GANs
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
> fractional  adj. 部分的；[数] 分数的，小数的 \
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.


# DETAILS OF ADVERSARIAL TRAINING
We trained DCGANs on three datasets, Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset. Details on the usage of each of these datasets are given below. No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1]. All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02. In the LeakyReLU, the slope of the leak was set to 0.2 in all models. While previous GAN work has used momentum to accelerate training, we used the Adam optimizer (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead. Additionally, we found leaving the momentum term β 1 at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.










