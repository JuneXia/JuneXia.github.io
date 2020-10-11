---
title: 
date: 2020-9-14
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

<!--more-->

先看代码示例：

```python
"""
References: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_ids = [7, 9]

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)

        # ************************************************
        # NOTE: 多卡训练时，forward中创建tensor该to到哪块卡上：
        # ************************************************
        # noise = Variable(torch.ones(output.shape))  # 默认在cpu上
        # noise = Variable(torch.ones(output.shape)).type(torch.cuda.FloatTensor)  # 默认to到cuda0
        # noise = Variable(torch.ones(output.shape)).cuda()  # 默认to到cuda0
        # noise = Variable(torch.ones(output.shape)).cuda(gpu_ids[0])  # 当只有gpu_ids中只有一块卡时是ok的，当gpu_ids有多块卡时失败，因为多卡是并行计算的，都to到一块卡上显然不行
        noise = Variable(torch.ones(output.shape)).cuda(output.device)  # ok
        # ************************************************

        output += noise
        print("\tIn Model: input size", input.size(),
              "output size", output.size(),
              "\tinput.device", input.device,
              "output.device", output.device,
              "\tnoise.device", noise.device)

        # NOTE: 返回值
        # return output  # 返回一个tensor OK
        # return output, noise  # 返回多个tensor OK
        # return output, {'output': output, 'noise': noise}  # 返回值有tensor，也有dict，OK
        # return output, {'image': {'output': output, 'noise': noise}}  # 返回的dict具有多层包装，OK
        return output, {'image': {'output': output, 'noise': noise}, 'scalar': noise.sum()}  # 返回的dict具有多层包装，OK

torch.backends.cudnn.benchmark = True
cudnn.benchmark = False

model = Model(input_size, output_size)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model, device_ids=gpu_ids)

model.to(gpu_ids[0])

for data in rand_loader:
    # input = data.to(gpu_ids[0])  # ok
    input = data  # ok
    output, noise = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size(),
          "output.device", output.device)

```


## 补充1：PyTorch中的tensor默认都是to到cuda0的，那么如何让其to到其他cuda呢？


**方法1: to 或 cuda 方法**

```python
gpu_ids = [7, 9]
tensor.to(gpu_ids[0]) 或者 tensor.cuda(gpu_ids[0])
```

**方法2: 借助其他已知tensor的device**

target_device = 通过如果是位于forward方法里面，可以通过取出forward中的其他tensor的当前device来作为目标device，参考上述代码示例；
```python
tensor.to(target_device)
```


**方法3：设置 cuda 环境变量**

```python
gpu_ids = [7, 9]

gpus = str()
for gpuid in gpu_ids:
    gpus += str(gpuid) + ','
gpus = gpus[0:-len(',')]
gpu_ids = np.arange(0, len(gpu_ids)).tolist()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])
```

添加上述代码后，此时 device[0] 指向的是 gpu7，device[1] 指向的是 gpu9，
也就是说： `tensor.to()、tensor.to([0])` 实际上都是将tensor to 到了gpu7上，注意这时候执行`tensor.to([7])`就会出错。


## forward 返回值
当采用nn.DataParallel做多卡训练时，返回值可以必须是位于cuda上的tensor。 \
（单卡训练则无此要求）


# 参考文献
[1] [Tutorials > Optional: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
[2] [Pytorch的nn.DataParallel](https://zhuanlan.zhihu.com/p/102697821)
[3] [pytorch 多GPU训练总结（DataParallel的使用）](https://blog.csdn.net/weixin_40087578/article/details/87186613)

