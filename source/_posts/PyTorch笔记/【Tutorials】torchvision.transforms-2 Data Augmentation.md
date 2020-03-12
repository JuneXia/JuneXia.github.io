---
title: 
date: 2019-09-20
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要介绍PyTorch中的数据增强(Data Augmentation)方法。
<!-- more -->

> torchvision.transforms : 常用的图像预处理方法
> - 数据中心化
> - 数据标准化
> - 缩放
> - 裁剪
> - 旋转
> - 翻转
> - 填充
> - 噪声添加
> - 灰度变换
> - 线性变换
> - 仿射变换
> - 亮度、饱和度及对比度变换

关于数据标准化上一节中已经讲过，本节主要讲述transforms中的数据增强方法。

&emsp; **数据增强**(Data Augmentation)又称为数据增广、数据扩增，它是对训练集进行变换，使训练集更丰富，从而让模型更具**泛化能力**。

> Transforms Methods一览:
> 
> 一、裁剪
> 1. transforms.CenterCrop
> 2. transforms.RandomCrop 
> 3. transforms.RandomResizedCrop
> 4. transforms.FiveCrop
> 5. transforms.TenCrop
> 
> 二、翻转和旋转
> 1. transforms.RandomHorizontalFlip
> 2. transforms.RandomVerticalFlip
> 3. transforms.RandomRotation
> 
> 三、图像变换
> 1. transforms.Pad
> 2. transforms.ColorJitter
> 3. transforms.Grayscale
> 4. transforms.RandomGrayscale
> 5. transforms.RandomAffine
> 6. transforms.LinearTransformation
> 7. transforms.RandomErasing
> 8. transforms.Lambda
> 9.  transforms.Resize
> 10. transforms.ToTensor
> 11. transforms. Normalize
> 
> 四、transforms的操作
> 1. transforms.RandomChoice
> 2. transforms.RandomApply
> 3. transforms.RandomOrder


# 裁剪

## transforms.CenterCrop
```python
class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
**功能**: 从图像中心裁剪图片
- **size**: 所需裁剪图片尺寸，可为int数值或长度为2的tuple.


**Code Examples:**
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    transforms.CenterCrop(50),  # 裁剪之后的尺寸是 50x50
    transforms.CenterCrop(512),  # CenterCrop输出尺寸比输入尺寸大，则在外围填充0
    transforms.CenterCrop((196, 256)),  # 也可输入元祖，表示 (target_height, target_width)

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


## transforms.RandomCrop
```python
class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        ...

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        ...
        
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        ...
        img = F.pad(img, ...)
        ...
        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
```
**功能**: 从图片中随机裁剪出尺寸为 size 的图片
- **size**: 所需裁剪图片尺寸
- **padding**: 设置填充大小 \
&emsp; &emsp; 当为 a 时, 上下左右均填充 a 个像素 \
&emsp; &emsp; 当为(a, b)时, 左右填充 a 个像素, 上下填充 b 个像素 \
&emsp; &emsp; 当为(a, b, c, d) 时, 左、上、右、下分别填充 a、b、c、d
- **pad_if_need**: 若图像小于设定的 size, 则填充
- **padding_mode**: 填充模式, 有4种模式 \
&emsp; &emsp; 1. constant: 像素值由 fill 设定 \
&emsp; &emsp; 2. edge: 像素值由图像边缘像素决定 \
&emsp; &emsp; 3. reflect: 镜像填充, 最后一个像素不镜像, eg : [1, 2, 3, 4] → [3, 2, 1, 2, 3, 4, 3, 2] \
&emsp; &emsp; 4. symmetric: 镜像填充, 最后一个像素镜像, eg : [1, 2, 3, 4] → [2, 1, 1, 2, 3, 4, 4, 3] \
- **fill**: constant 时, 设置填充的像素值


## transforms.RandomResizedCrop
```python
class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        ...

        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        ...
        return format_string

```

**功能**: 随机大小、长宽比裁剪图片。
- **size**: 所需裁剪图片尺寸
- **scale**: 随机裁剪面积比例, 默认(0.08, 1), 例如scale=0.6时，表示对输入图片的60%进行裁剪
- **ratio**: 随机长宽比, 默认(3/4, 4/3)，即长宽比最大比例为3/4或者4/3
- **interpolation**: 插值方法  \
&emsp; &emsp; PIL.Image.NEAREST，最近邻插值 \
&emsp; &emsp; PIL.Image.BILINEAR，双线性插值 \
&emsp; &emsp; PIL.Image.BICUBIC，双三次插值 \

**具体操作方法就是**：scale和ratio都是一个tuple, 先在scale取值范围内选取一个裁剪面积比例 $scale_i$、在ratio取值范围之内选择一个长宽比比例$ratio_i$，然后根据这个长宽比$ratio_i$从输入图片中裁剪出面积为$scale_i$的图片来，最后再resize到target_size大小。

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 3 RandomResizedCrop
    transforms.RandomResizedCrop(size=512, scale=(0.3, 0.9)),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


## transforms.FiveCrop
```python
class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
**功能**: 在图像的上下左右以及中心裁剪出尺寸为 size 的5张图片.
- **size**: 所需裁剪图片尺寸


## transforms.TenCrop
```python
class TenCrop(object):
    """Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)
```
**功能**: 内部会调用FiveCrop操作，对FiveCrop裁剪得到的5张图片进行水平或者垂直镜像获得 10 张图片.
- **size**: 所需裁剪图片尺寸
- **vertical_flip**: 是否垂直翻转

**注意**：假设输入shape为[b, 3, 224, 224]，对于每一张图片，FiveCrop返回的长度为5的tuple，TenCrop返回的是长度为10的tuple，所以通过使用FiveCrop或者TenCrop的transforms.Compose([...])后，返回的shape应该是[b, 5, 3, 224, 224]或者[b, 10, 3, 224, 224].

**FiveCrop、TenCrop代码示例**:
```python

...

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 4 FiveCrop
    transforms.FiveCrop(112),  # FiveCrop 返回的是一个tuple，无法被后续的transforms所接收，所以这里需要对其做拼接。
    transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    transforms.TenCrop(112, vertical_flip=False),
    transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 上面已经做过ToTensor操作了，所以这里就不需要再做ToTensor啦。
    # TODO: 但是为什么连transforms.Normalize也不能做了呢？Normalize会出错。
    # transforms.ToTensor(),
    # transforms.Normalize(norm_mean, norm_std),
])

...

# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W

        # 返回单个图片的可视化方法
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # img_tensor = inputs[0, ...]     # C H W
        # img = transform_invert(img_tensor, train_transform)
        # plt.imshow(img)
        # plt.show()
        # plt.pause(0.5)
        # plt.close()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # FiveCrop、TenCrop等返回多个图片的可视化方法
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        bs, ncrops, c, h, w = inputs.shape
        for n in range(ncrops):
            img_tensor = inputs[0, n, ...]  # C H W
            img = transform_invert(img_tensor, train_transform)
            plt.imshow(img)
            plt.show()
            plt.pause(1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```


# 翻转和旋转

## transforms.RandomHorizontalFlip
```python
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
```
**功能**: 依概率水平(左右)翻转图片
- **p**: 翻转概率


## transforms.RandomVerticalFlip
```python
class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
```
**功能**: 依概率垂直(上下)翻转图片
- **p**: 翻转概率

代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 Horizontal Flip
    transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    transforms.RandomVerticalFlip(p=0.5),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


## transforms.RandomRotation
```python
class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        ...

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        ...
        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        ...
        return format_string
```
**功能**: 随机旋转图片

- **degrees**: 旋转角度
&emsp; &emsp; 当为a时, 在(-a, a)之间选择旋转角度
&emsp; &emsp; 当为(a, b)时, 在(a, b)之间选择旋转角度
- **resample**: 重采样方法, 通常采用默认值即可
- **expand**: 是否扩大图片, 以保持原图信息
- **center**: 旋转点设置, 即围绕着这个点进行旋转，默认为中心旋转

**注意**：
1. expand=True时，每次得到的图片大小是不一样的。当batchsize>1且expand=True时，则一个batchsize中可能会存在多个不同尺寸的图像，则这时候需要自定义一个方法将这些不同的size的图片都resize到同一尺寸，否则程序运行时会报错；
2. 当expand=True时，如果RandomRotation是围绕中心点进行旋转，则图片不会丢失信息，而如果RandomRotation是围绕左上角点(0, 0)进行旋转，则图片还是会丢失信息，这一点可参见图1、图2所示。


<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomrotation1.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomrotation2.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图1 &nbsp; 围绕图片中心旋转(左边expand=False, 右边expand=True)</center>
</html>

<br>

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomrotation3.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomrotation4.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图2 &nbsp; 围绕图片左上角(0, 0)旋转(左边expand=False, 右边expand=True)</center>
</html>

代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 3 RandomRotation
    transforms.RandomRotation(90),
    transforms.RandomRotation((90), expand=True),
    transforms.RandomRotation(30, center=(0, 0)),
    transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


# 图像变换
## transforms.Pad
```python
class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image

            - reflect: pads with reflection of image without repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)
```
**功能**: 对图片边缘进行填充
**padding**: 设置填充大小
&emsp; &emsp; 当为a时, 上下左右均填充 a 个像素
&emsp; &emsp; 当为(a, b)时,上下填充 b 个像素, 左右填充 a 个像素
&emsp; &emsp; 当为(a, b, c, d)时, 表示左、上、右、下分别填充 a、b、c、d
- **padding_mode**: 填 充模 式 ,有 4 种模式, constant、edge、reflect 和 symmetric
- **fill**: 当padding_mode为constant时, 设置填充的像素值, (R, G, B) or (Gray). 当padding_mode为其他模式时，fill参数无效。

代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 Pad
    transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

## transforms.ColorJitter
```python
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        ...
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        ...

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        ...
        return format_string
```
**功能**: 调整亮度、对比度、饱和度和色相
**brightness**: 亮度调整因子
&emsp; &emsp; 当为a时, 从[max(0, 1-a), 1+a]中随机选择
&emsp; &emsp; 当为(a, b)时, 从[a, b]中
- **contrast**: 对比度参数, 同brightness
- **saturation**: 饱和度参数, 同brightness
- **hue**: 色相参数, 当为 a 时, 从[-a, a]中选择参数, 注: 0 <= a <= 0.5
&emsp; &emsp; &emsp; 当为(a, b)时, 从[a, b]中选择参数 , 注: -0.5 <= a <= b <= 0.5

代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 2 ColorJitter
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ColorJitter(hue=0.3),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


## transforms.Grayscale
```python
class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)
```
**功能**: 将图片转换为灰度图
- **num_ouput_channels**: 输出通道数，只能设1或3


## transforms.RandomGrayscale
```python
class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)
```
**功能**: 依概率将图片转换为灰度图
- **p**: 概率值, 图像被转换为灰度图的概率。如果设置p=1.0，则就等价于Grayscale了。

代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 3 Grayscale
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomGrayscale(p=0.5),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

## transforms.RandomAffine
```python
class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        ...

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        ...
        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        ...
        return s.format(name=self.__class__.__name__, **d)
```
**功能**: 对图像进行仿射变换, 仿射变换是二维的线性变换, 由五种基本原子变换构成, 分别是旋转、平移、缩放、错切和翻转，经过这五种变换的随机组合就可以得到二维的线性变换。
- **degrees**: 旋转角度设置
- **translate**: 平移区间设置, 如(a, b), a 设置宽(width), b 设置高(height)图像在宽维度平移的区间为 -img_width * a < dx < img_width * a
- **scale**: 缩放比例(以面积为单位)
- **fill_color**: 填充颜色设置
- **shear**: 错切角度设置, 有水平错切和垂直错切
&emsp; &emsp; 若为a, 则仅在 x 轴错切, 错切角度在(-a, a)之间
&emsp; &emsp; 若为(a, b), 则a为x轴错切角度, b为y轴错切角度
&emsp; &emsp; 若为(a, b, c, d), 则x轴的角度范围为(a, b), y轴的错切角度为(c, d)
- **resample**: 重采样方式, 有NEAREST、BILINEAR、BICUBIC

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomaffine_shear1.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomaffine_shear2.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图3 &nbsp; 仿射变换之错切(左边为在y轴错切, 右边为在x轴错切)</center>
</html>

代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 4 Affine
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    transforms.RandomAffine(degrees=0, shear=(0, 45, 0, 0)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


## transforms.RandomErasing
```python
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        ...

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        ...
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            return F.erase(img, x, y, h, w, v, self.inplace)
        return img
```
**功能**: 对图像进行随机遮挡

- **p**: 概率值, 执行该操作的概率
- **scale**: 遮挡区域的面积
- **ratio**: 遮挡区域长宽比
- **value**: 设置遮挡区域的像素值, Default is 0. If a single int, it is used to erase all pixels. If a tuple of length 3, it is used to erase R, G, B channels respectively. If a str of 'random', erasing each pixel with random values.
参考文献:《Random Erasing Data Augmentation》

**注意**：transforms.RandomErasing 接受的是一个Tensor，而其他的transforms接受的是一个PIL图片。

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 5 Erasing
    transforms.ToTensor(),
    # RandomErasing之前的ToTensor已经将数据范围设置到[0.0, 1.0]了，所以这里填充的value也应是[0.0, 1.0]范围的值，否则就不能正确设置指定的颜色填充。
    transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),

    # RandomErasing的返回已经是一个Tensor了，所以就不需要再ToTensor了。ToTensor接受的是一个PIL图片或者numpy.ndarray
    # transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


## transforms.Lambda
```python
class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
```

**功能**: 用户自定义 lambda 方法
- **lambd**: lambda 匿名函数
lambda [arg1 [,arg2, ... , argn]]: expression


代码示例在FiveCrop中已经有所展示了，这里就不具体再说了。
```python
...

transforms.FiveCrop(112),  # FiveCrop 返回的是一个tuple，无法被后续的transforms所接收，所以这里需要对其做拼接。
transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

...
```


# transforms组合操作
&emsp; 对一组transforms操作进行组合应用，主要有三种组合方式，下面将一一介绍。

## transforms.RandomChoice
```python
class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)
```
**功能**: 从一系列transforms方法中随机挑选一个\
`transforms.RandomChoice([transforms1, transforms2, transforms3])`


## transforms.RandomApply
```python
class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        ...
        return format_string
```
**功能**: 依据概率执行一组transforms操作\
`transforms.RandomApply([transforms1, transforms2, transforms3], p=0.5)`


## transforms.RandomOrder
```python
class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img
```
**功能**: 对一组transforms操作打乱顺序\
`transforms.RandomOrder([transforms1, transforms2, transforms3])`


RandomChoice、RandomApply、RandomOrder 代码示例：
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 RandomChoice
    transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
                            transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
    transforms.RandomOrder([transforms.RandomRotation(15),
                            transforms.Pad(padding=32),
                            transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


# User-Defined Transforms

```python
class Compose(object):
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
```
&emsp; PyTorch中的transforms方法都是在transforms.Compose方法中被调用的，通过观察Compose中的调用方法，可以总结出自定义transforms的一些要素。

自定义transforms的要素:
1. 仅接收一个参数, 返回一个参数
2. 注意上下游的输出与输入

通过类实现多参数传入:
```python
class YourTransforms(object):
    def __init__(self, ...):
        ...
    
    def __call__(self, img):
        ...
    return img
```

下面以依概率生成椒盐噪声为例，实现自定义的transforms方法。

&emsp; **椒盐噪声**又称为脉冲噪声, 是一种随机出现的白点或者黑点, 白点称为盐噪声, 黑色为椒噪声。
**信噪比**(Signal-Noise Rate, SNR)是衡量噪声的比例, 图像中为图像像素的占比。

<div align=center>
    <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_User-Defined-Transforms1.jpg" width = 100% height = 100% />
</div>
<center>图4 &nbsp; 具有不同信噪比的椒盐噪声</center>


代码示例：
```python
# -*- coding: utf-8 -*-
"""
# @file name  : my_transforms.py
# @author     : tingsongyu
# @date       : 2019-09-13 10:08:00
# @brief      : 自定义一个transforms方法
"""
import os
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tools.my_dataset import RMBDataset
from tools.common_tools import transform_invert


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# ============================ step 1/5 数据 ============================
split_dir = os.path.join("..", "..", "data", "rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    AddPepperNoise(0.9, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W

        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
```


# 数据增强实战应用

## 数据增强的原则

**原则**: 让训练集与测试集更接近
- **空间位置**: 平移
- **色彩**: 灰度图, 色彩抖动
- **形状**: 仿射变换
- **上下文场景**: 遮挡, 填充
- ......

<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_data_augmentation_application2.jpg" width = 50% height = 50% />
                </div>
            </td>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_data_augmentation_application1.jpg" width = 50% height = 50% />
                </div>
            </td>
        </tr>
    </table>
    <center>图4 &nbsp; 数据增强的原则</center>
</html>


## 数据增强案例
<div align=center>
    <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_data_augmentation_application3.jpg" width = 50% height = 50% />
</div>
<center>图5 &nbsp; RMB识别任务</center>

&emsp; 如图5所示，我们使用第四套人民币为训练集，以第五套人民币为测试集。我们使用下面的数据增强方法。

```python
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

&emsp; 如果我们仅仅使用上面的这种数据增强的话，则训练出来的模型很可能将第五套人民币的100元预测成1元，因为它们的颜色更加接近。鉴于此，可以在数据增强中将人民币的颜色信息给去掉，即随机将图片转为灰度图，改进后的数据增强代码如下。

```python
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.9),  # 随机将图片转为灰度图
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```


# 参考文献
[1] DeepShare.net > PyTorch框架

