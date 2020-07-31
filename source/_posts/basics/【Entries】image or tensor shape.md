---
title: 
date: 2020-7-27
tags:
categories: ["basics"]
mathjax: true
---

```python
impath = './tmp.jpg'
img_pil = PIL.Image.Image.Open(impath)
w, h = img_pil.size

img_cv = cv2.imread(impath)
h, w, c = img_cv.shape

tensor = torchvision.transforms.ToTensor()(img_pil)
c, h, w = tensor.shape


```





