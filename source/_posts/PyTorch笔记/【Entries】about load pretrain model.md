---
title: 
date: 2020-9-22
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

<!--more-->

# PyTorch恢复训练时，加载优化器参数会出现device不一致的错误

示例代码如下：

```python
gpu_ids = [5]

# 定义network
net = BackgroundMattingModel(config, pretrain_model=pretrain_model)

# 定义optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)
optimizer.load_state_dict(torch.load(pretrain_optim))
# 默认会将一部分参数加载到保存 pretrain_optim 时所在的gpu卡上 并且无法被释放，但是实际上我们本次恢复训练时并不一定还会用到之前的卡，这会导致一部分GPU显存被浪费；
# 而用下面这条语句则能够解决该问题。
# optimizer.load_state_dict(torch.load(pretrain_optim, map_location=torch.device(gpu_ids[0])))

# network 并行化
net = nn.DataParallel(net, device_ids=gpu_ids).to(gpu_ids[0])  # 有无环境变量设置均可，forward传递tensor ok;

# 迭代训练
for i, data in enumerate(train_loader):
    loss, state_dict = net(data, 'TRAINING')
    
    optimizer.zero_grad()
    loss = loss.mean()
    loss.backward()
    optimizer.step()
```

上述代码的最后一行 `optimizer.step()` 执行时会报错如下：
```python
Traceback (most recent call last):
  File "/home/xxx/.pycharm_helpers/pydev/pydevd.py", line 1438, in _exec
    pydev_imports.execfile(file, globals, locals)  # execute the script
  File "/home/xxx/.pycharm_helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/home/xxx/promt/matting/train_mt_adobe_v2.py", line 303, in <module>
    optimizer.step()
  File "/home/xxx/.conda/envs/py37pt1.4/lib/python3.7/site-packages/torch/optim/adam.py", line 95, in step
    exp_avg.mul_(beta1).add_(1 - beta1, grad)
RuntimeError: expected device cpu but got device cuda:5
```

实验发现如果将代码中optimizer加载参数的语句 `optimizer.load_state_dict(torch.load(pretrain_optim))` 注释掉则程序可以正常运行，而加上这条语句代码则执行失败。

参考文献[1]，在optimizer load完参数后，增加如下代码即可解决：

```python
gpu_ids = [5]

# 定义network
net = ...

# 定义optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)
optimizer.load_state_dict(torch.load(pretrain_optim))
# optimizer.load_state_dict(torch.load(pretrain_optim, map_location=torch.device(gpu_ids[0])))
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(gpu_ids[0])

# network 并行化

...

```



**分析原因：** \

`optimizer = optim.Adam(net.parameters(), lr=1e-4)` 中加载的只是跟 network 相关的网络参数，这些参数是被存储在 optimizer.param_groups 中的，而 optimizer.state 中存储的是历史训练状态，如BN的 alpha、beta 等参数。
- 如果不加载 optimizer 的历史参数，则 optimizer.state 中的历史参数为空，optimizer会自动在运行时记录这些训练参数（BN），此时这些参数所在device是随网络输出tensor所在device的，所以训练不会出错；
- 如果加载 optimizer 的历史参数，则加载到 optimizer.state 中的历史参数默认会被存放在 cpu 上的，这就导致了上述训练时 device 不一致的错误；所以需要手动将 optimizer.state 中的参数都 to 到目标 device 上。如果是多卡训练也是 to 到多卡中的主卡上.


# 参考文献
[1] [optimizer load_state_dict() problem?](https://github.com/pytorch/pytorch/issues/2830)

