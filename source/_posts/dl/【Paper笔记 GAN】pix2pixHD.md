---
title: 
date: 2020-6-26
tags:
categories: ["深度学习笔记"]
mathjax: true
---

[High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs]() \
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro \
NVIDIA Corporation, UC Berkeley \
2018 CVPR


**Abstract**
&emsp; We present a new method for synthesizing high-resolution photo-realistic images from semantic label maps using conditional generative adversarial networks (conditional GANs). Conditional GANs have enabled a variety of applications, but the results are often limited to low-resolution and still far from realistic. In this work, we generate 2048 × 1024 visually appealing results **with a novel adversarial loss, as well as new multi-scale generator and discriminator architectures**. Furthermore, we **extend our framework to interactive(adj. 交互式的；相互作用的) visual manipulation with two additional features**. First, we **incorporate object instance segmentation information**, which enables object manipulations such as removing/adding objects and changing the object category. Second, we **propose a method to generate diverse(adj. 不同的，相异的；多种多样的，形形色色的) results given the same input**, allowing users to edit the object appearance interactively. Human opinion(n. 意见；主张) studies demonstrate that our method significantly outperforms existing methods, `advancing both the quality and the resolution of deep image synthesis(n.综合,[化学]合成) and editing. (提高了图像深度合成和编辑的质量和分辨率.)`
> summary: 我们提出了一种新方法来从语义标签图来合成高分辨率的真实图片的新方法，该方法使用conditional GANs。我们使用一个新颖的adversarial-loss 和新的多尺度网络结构来生成 2048 × 1024 的图片。此外，我们还使用两个额外的feature来扩展我们的框架来进行交互式视觉操作。第一，我们合并目标实例分割信息，这将使得像“增加或移除目标”、“改变目标种类”这类操作变得可行。第二，我们提出了一个方法来从相同的输入生成不同的结果，允许用户交互式编辑目标外观。


# Introduction
&emsp; Rendering(n.渲染;表现;表演;描写;翻译) photo-realistic images using standard graphics techniques is involved, since geometry(n.几何学), materials, and light transport must be simulated explicitly(adv.明确地;明白地). Although existing graphics algorithms excel at the task, building and editing virtual environments is expensive and time-consuming. That is because we have to model every aspect(n.方面;方向;形势;外貌) of the world explicitly. `If we were able to render photo-realistic images using a model learned from data, we could turn the process of graphics rendering into a model learning and inference problem. (如果我们能够使用从数据中学习到的模型来渲染逼真的图像，我们就可以将图形渲染的过程转化为模型学习和推理问题.)` Then, we could simplify the process of creating new virtual worlds by training models on new datasets. We could even make it easier to customize environments by allowing users to simply specify semantic information rather than modeling geometry, materials, or lighting.
> summary: 图像渲染所涉及到的技术很复杂，如果我们能够使用从数据种学习到的模型来渲染图像，那么我们便可以将图形渲染过程转化为一个模型学习和推理过程。

&emsp; In this paper, we discuss a new approach that produces high-resolution images from semantic label maps. This method has a wide range of applications. For example, we can use it to create synthetic(adj.综合的;合成的,人造的) training data for training visual recognition algorithms, `since it is much easier to create semantic labels for desired scenarios than to generate training images (因为为所需场景创建语义标签要比生成训练图像容易得多).` Using semantic segmentation methods, **we can transform images into a semantic label domain, edit the objects in the label domain, and then transform them back to the image domain.** This method also gives us new tools for higher-level image editing, e.g., adding objects to images or changing the appearance of existing objects.
> summary: 本文我们将讨论一种新的从语义标签map来产生高分辨率图像的新方法，这种方法有很多应用场景，如生成训练样本、图像编辑。


&emsp; To synthesize(v.合成;综合) images from semantic labels, one can use the pix2pix method, an image-to-image translation framework [21] which leverages(n. 手段，影响力；杠杆作用; v. 利用；举债经营) generative adversarial networks (GANs) [16] in a conditional setting. Recently, Chen and Koltun [5] suggest that adversarial training might be unstable and prone(adj.俯卧的;有…倾向的,易于…的) to failure for high-resolution image generation tasks. `Instead(替代的,相反的), they adopt a modified perceptual loss [11, 13, 22] to synthesize images, which are highresolution but often lack fine details and realistic textures. (相反，他们采用一种改进的感知损失来合成高分辨率图片，但是这些图片经常缺乏较好的细节以及逼真的纹理.)`
> summary: 在通过语义标签合成图像方面，有的人使用pix2pix的conditional-GAN方法。但最近 Chen 和 Koltun 等人认为GAN不稳定且在生成高分辨率图像时容易失败，所以与使用GAN方法相反，他们使用一种改进的感知损失来合成高分辨率图片，但也还是有缺乏细节纹理等问题。


&emsp; Here we address two main issues of the `above(超过;高于;在…上面;前文)` state-of-the-art methods: (1) the difficulty of generating highresolution images with GANs [21] and (2) the lack of details and realistic textures in the previous high-resolution results [5]. We show that through a new, robust adversarial learning objective together with new **multi-scale generator and discriminator architectures**, we can synthesize photo-realistic images at 2048 × 1024 resolution, which are more visually appealing than those computed by previous methods [5,21]. We **first** obtain our results with adversarial training only, without relying on any hand-`crafted (adj.精心制作的)` losses [44] or pre-trained networks (e.g. VGGNet [48]) for perceptual losses [11,22] (Figs. 7c, 9b). **Then** we show that **adding perceptual losses from pre-trained networks [48] can slightly(adv. 些微地，轻微地；纤细地，瘦小的) improve the results in some circumstances (Figs. 7d, 9c)**, if a pre-trained network is available. `Both results outperform previous works substantially(adv.实质上;大体上;充分地) in terms of image quality. (在图像质量方面，这两个结果都超过了之前的工作).`
> summary: 这里我们解决上述先进方法的两个主要问题：(1) GANs生成高分辨率图片很困难；(2) 在生成的高分辨率图片时缺乏细节以及逼真的纹理。我们证明了通过使用一个新的对抗学习目标以及多尺度网络结构 能够有效合成高分辨率图片，如果再增加一个预训练网络的感知损失则可以再提高些许结果。


&emsp; Furthermore, to support interactive semantic manipulation, we extend our method in two directions. First, we **use instance-level object segmentation information, which can separate different object instances within the same category**. This enables flexible object manipulations, such as adding/removing objects and changing object types. Second, we **propose a method to generate diverse(adj. 不同的，相异的；多种多样的，形形色色的) results given the same input label map**, allowing the user to edit the appearance of the same object interactively. 


&emsp; We compare against state-of-the-art visual synthesis systems [5, 21], and show that our method outperforms these approaches regarding both quantitative evaluations and human perception studies. We also perform an ablation study regarding(prep.关于,至于;就…而论) the training objectives and the importance of instance-level segmentation information. Our code and data are available at our website.
> summary: 我们和最先进的视觉合成系统[5,21]作比较，结果证明我们的方法要更好。


# Related Work
&emsp; **Generative adversarial networks** Generative adversarial networks (GANs) [16] aim to model the natural image distribution by forcing the generated samples to be indistinguishable from natural images. GANs enable a wide variety of applications such as image generation [1, 42, 62], representation learning [45], image manipulation [64], object detection [33], and video applications [38, 51, 54]. Various coarse-to-fine schemes [4] have been proposed [9,19,26,57] to synthesize larger images (e.g. 256 × 256) in an unconditional setting. Inspired by their successes, **we propose a new coarse-to-fine generator and multi-scale discriminator architectures** suitable for conditional image generation at a much higher resolution.
> summary: 受 [4,9,19,26,57] 启发，我们提出了一个新的 coarse-to-fine 生成器和多尺度鉴别器结构，适合更高分辨率下的条件图像生成。


&emsp; **Image-to-image translation** Many researchers have leveraged(leverage n.手段,影响力;杠杆作用;杠杆效率;v.利用;举债经营) adversarial learning for image-to-image translation [21], whose goal is to translate an input image from one domain to another domain given input-output image pairs as training data. Compared to L1 loss, which often leads to blurry images [21, 22], the adversarial loss [16] has become a popular choice for many image-to-image tasks [10, 24, 25, 32, 41, 46, 55, 60, 66]. The reason is that the **discriminator can learn a trainable loss function** and automatically adapt to the differences between the generated and real images in the target domain. For example, the recent pix2pix framework [21] used image-conditional GANs [39] for different applications, such as transforming Google maps to satellite views and generating cats from user sketches. Various methods have also been proposed to learn an image-to-image translation in the absence(n. 没有；缺乏；缺席；不注意) of training pairs [2, 34, 35, 47, 50, 52, 56, 65].
> summary: 和L1损失会产生模糊的图片相比，对抗损失在一些 image-to-image 任务上逐渐变成一个流行的选择，因为discriminator会学习一个可训练的损失函数。（discriminator本省可以看成一个损失函数，只不过这个损失函数是一个网络结构.）


&emsp; Recently, Chen and Koltun [5] suggest that it might be hard for conditional GANs to generate high-resolution images due to the training instability and optimization issues. To avoid this difficulty, they use a `direct regression objective (直接回归目标)` based on a perceptual loss [11, 13, 22] and produce the first model that can synthesize 2048 × 1024 images. The generated results are high-resolution but often lack fine details and realistic textures. Our method is motivated by their success. We show that **using our new objective function as well as novel multi-scale generators and discriminators**, we not only largely stabilize the training of conditional GANs on high-resolution images, but also achieve significantly better results compared to Chen and Koltun [5]. `Side-by-side (adv.并肩地;并行地)` comparisons clearly show our advantage(n.优势;利益;有利条件;vt.获利;有利于;使处于优势) (Figs. 1, 7, 8, 9).
> summary: 针对使用 conditional GANs 来生成高分辨率图像所面临的训练不稳定以及优化问题，Chen and Koltun [5] 等人使用基于感知损失来直接回归目标的方法 是第一个能够合成2048×1024图片的模型，但是他们合成的图片经常会缺乏较好的细节以及逼真的纹理。受他们的启发，我们这里使用一个新的目标函数和新颖的多尺度generator和discriminator，我们不仅能够使训练稳定，还能够产生比Chen and Koltun更好的结果。


&emsp; **Deep visual manipulation** Recently, deep neural networks have obtained promising results in various image processing tasks, such as style transfer [13], inpainting(图像修复) [41], colorization [58], and restoration [14]. However, most of these works lack an interface for users to adjust the current result or explore the output space. To address this issue, Zhu et al. [64] developed an optimization method for editing the object appearance based on the priors learned by GANs. Recent works [21, 46, 59] also provide user interfaces for creating novel imagery from low-level cues such as color and sketch. All of the prior works report results on low-resolution images. Our system shares the same spirit(n. 精神；心灵；情绪；志气; vt. 鼓励；鼓舞；诱拐) as this past work, but we focus on object-level semantic editing, allowing users to interact with the entire scene and manipulate individual objects in the image. As a result, users can quickly create a novel scene with minimal effort(n.努力;成就). Our interface is inspired by prior data-driven graphics systems [6, 23, 29]. But our system allows more flexible(adj. 灵活的；柔韧的；易弯曲的) manipulations and produces high-res results in real-time.
> summary: 最近，深度学习在很多图像处理任务上取得了成功，然而这些工作都缺乏一个使用户能够调整当前结果或者探索输出空间的接口。就算有这种交互接口，但也都是从一些诸如颜色、素描这类低级别的接口线索来创建新颖的图片，而且之前产生的都是一些低分辨率的图片。我们受前人的些许启发，但是我们更关注目标级别的语义编辑。


# Instance-Level Image Synthesis
&emsp; We propose a conditional adversarial framework for generating high-resolution photo-realistic images from semantic label maps. We first review our baseline model pix2pix (Sec. 3.1). We then describe how we increase the photo-realism and resolution of the results with our improved objective function and network design (Sec. 3.2). Next, we use additional instance-level object semantic information to further improve the image quality (Sec. 3.3). Finally, we introduce an instance-level feature embedding scheme to better handle the multi-modal nature of image synthesis, which enables interactive object editing (Sec. 3.4).


## The pix2pix Baseline
&emsp; The pix2pix method [21] is a conditional GAN framework for image-to-image translation. It consists of a generator G and a discriminator D. For our task, the objective of the generator G is to translate semantic label maps to realistic-looking images, while the discriminator D aims to distinguish real images from the translated ones. The framework operates in a supervised setting. In other words, the training dataset is given as a set of pairs of corresponding images ${(s_i, x_i)}$, where $s_i$ is a semantic label map and $x_i$ is a corresponding natural photo. **Conditional GANs aim to model the conditional distribution of real images given the input semantic label maps via the following minimax game:**
$$
\min \limits_{G} \max \limits_{D} \mathcal{L}_{GAN} (G, D)  \tag{1}
$$

where the objective function $\mathcal{L}_{GAN} (G, D)$^1 is given by
$$
\mathbb{E}_{s, x} [ \text{log} D(s, x) ] + \mathbb{E}_s [ \text{log} (1 - D(s, G(s))) ].  \tag{2}
$$

> we denote $\mathbb{E}_s \triangleq \mathbb{E}_{s \sim p_{data} (s)}$ and $\mathbb{E}_{(s, x)} \triangleq \mathbb{E}_{(s, x) \sim p_{data} (s, x)}$ for simplicity.


The pix2pix method adopts U-Net [43] as the generator and a patch-based fully convolutional network [36] as the discriminator. The input to the discriminator is a channel-wise concatenation of the semantic label map and the corresponding image. However, the resolution of the generated images on Cityscapes [7] is up to 256 × 256. We tested directly applying the pix2pix framework to generate highresolution images, but found the training unstable and the quality of generated images unsatisfactory. We therefore describe how we improve the pix2pix framework in the next subsection.
> summary: pix2pix 方法输入到 discriminator 的是语义标签图和对应的真实图片之间的逐通道连接，然而它所生成图片只达到了256 × 256. 我们有测试直接使用 pix2pix 来生成高分辨率的图像，但发现这样做训练不稳定且生成质量欠佳。


&emsp; We improve the pix2pix framework by using a coarse-to-fine generator, a multi-scale discriminator architecture, and a robust adversarial learning objective function. 

**Coarse-to-fine generator** We decompose(vi.分解;腐烂) the generator into two sub-networks: G1 and G2. We term G1 as the global generator network and G2 as the local enhancer network. The generator is then given by the tuple G = {G1, G2} as visualized in Fig. 2. The global generator network operates at a resolution of 1024 × 512, and `the local enhancer network outputs an image with a resolution that is 4× the output size of the previous one (2× along each image dimension).`   \
局部增强网络以之前尺寸的4倍分辨率来输出图片. \
For synthesizing images at an even higher resolution, additional local enhancer networks could be utilized(utilize v.利用). For example, the output image resolution of the generator G = {G1, G2} is 2048 × 1024, and the output image resolution of G = {G1, G2, G3} is 4096 × 2048.
> 待总结。。。。。。。。。。，如果想要输出更高分辨率的图片，则可以继续附加 local enhancer network 来达到此目的。




Our global generator is built on the architecture proposed by Johnson et al. [22], which has been proven successful for neural style transfer on images up to 512 512. It consists of 3 components: a convolutional front-end G(F ) 1 , a set of residual blocks G(R) 1 [18], and a transposed convolutional back-end G(B) 1 . A semantic label map of resolution 1024 512 is passed through the 3 components sequentially to output an image of resolution 1024 512.


The local enhancer network also consists of 3 components: a convolutional front-end G(F ) 2 , a set of residual blocks G(R) 2 , and a transposed convolutional back-end G(B)^2. The resolution of the input label map to G2 is 2048 1024. Different from the global generator network, the input to the residual block G(R) 2 is the element-wise sum of two feature maps: the output feature map of G(F ) 2 , and the last feature map of the back-end of the global generator network G(B) 1 . This helps integrating the global information from G1 to G2.


During training, we first train the global generator and then train the local enhancer in the order of their resolutions. We then jointly fine-tune all the networks together. We use this generator design to effectively aggregate global and local information for the image synthesis task. We note that such a multi-resolution pipeline is a wellestablished practice in computer vision [4] and two-scale is often enough [3]. Similar ideas but different architectures could be found in recent unconditional GANs [9, 19] and conditional image generation [5, 57].


Multi-scale discriminators High-resolution image synthesis poses a great challenge to the GAN discriminator design. To differentiate high-resolution real and synthesized images, the discriminator needs to have a large receptive field. This would require either a deeper network or larger convolutional kernels. As both choices lead to an increased network capacity, overfitting would become more of a concern. Also, both choices require a larger memory footprint for training, which is already a scarce resource for highresolution image generation. 


To address the issue, we propose using multi-scale discriminators. We use 3 discriminators that have an identical network structure but operate at different image scales. We will refer to the discriminators as D1, D2 and D3. Specifically, we downsample the real and synthesized highresolution images by a factor of 2 and 4 to create an image pyramid of 3 scales. The discriminators D1, D2 and D3 are then trained to differentiate real and synthesized images at the 3 different scales, respectively. Although the discriminators have an identical architecture, the one that operates at the coarsest scale has the largest receptive field. It has a more global view of the image and can guide the generator to generate globally consistent images. On the other hand, the discriminator operating at the finest scale is specialized in guiding the generator to produce finer details. This also makes training the coarse-to-fine generator easier, since extending a low-resolution model to a higher resolution only requires adding an additional discriminator at the finest level, rather than retraining from scratch. Without the multi-scale discriminators, we observe that many repeated patterns often appear in the generated images.



With the discriminators, the learning problem in Eq. (1) then becomes a multi-task learning problem of
$$
\min \limits_{G} \max \limits_{D_1, D_2, D_3} \sum_{k=1,2,3} \mathcal{L}_{GAN} (G, D_k).  \tag{3}
$$

Using multiple GAN discriminators at the same image scale has been proposed in unconditional GANs [12]. Iizuka et al. [20] add a global image classifier to conditional GANs to synthesize globally coherent content for inpainting. Here we extend the design to multiple discriminators at different image scales for modeling high-resolution images. Improved adversarial loss We improve the GAN loss in Eq. (2) by incorporating a feature matching loss based on the discriminator. This loss stabilizes the training as the generator has to produce natural statistics at multiple scales. Specifically, we extract features from multiple layers of the discriminator, and learn to match these intermediate representations from the real and the synthesized image. For ease of presentation, we denote the ith-layer feature extractor of discriminator Dk as D(i) k (from input to the ith layer of Dk). The feature matching loss LFM(G, Dk) is then calculated as:
$$
\mathcal{L}_{FM} (G, D_k) = \mathbb{E}_{s, x} \sum^T_{i=1} \frac{1}{N_i} [ \Vert D_k^{(i)} (s, x) - D_k^{(i)} (s, G(s)) \Vert_1 ].  \tag{4}
$$

where T is the total number of layers and Ni denotes the number of elements in each layer. Our GAN discriminator feature matching loss is related to the perceptual loss [11, 13,22], which has been shown to be useful for image superresolution [32] and style transfer [22]. In our experiments, we discuss how the discriminator feature matching loss and the perceptual loss can be jointly used for further improving the performance. We note that a similar loss is used for training VAE-GANs [30].


Our full objective combines both GAN loss and feature matching loss as:
$$
\min \limits_{G} \bigg( \Big( \max \limits_{D_1, D_2, D_3} \sum_{k=1,2,3} \mathcal{L}_{GAN} (G, D_k) \Big) + \lambda \sum_{k=1,2,3} \mathcal{L}_{FM} (G, D_k) \bigg)  \tag{5}
$$


















