---
title: 
date: 2020-05-30
tags:
categories: ["开发环境与工具"]
mathjax: true
---
windows 下的 anaconda + pytorch 环境搭建.
<!-- more -->


**step1:** 安装 Anaconda
参考文献[1] 

(这里就安装好 Anaconda 就好，conda 的第三方软件包(如opencv、pytorch等)就不使用 Anaconda GUI 安装了，我们使用 Anaconda Prompt 命令行工具安装这些第三方包)

官网：https://www.anaconda.com/   \
最新版本下载地址：https://www.anaconda.com/download/   \
历史版本：https://repo.anaconda.com/archive/

我下载的是最新版：Anaconda3-2020.02-Windows-x86_64，
该版本默认使用的是python3.7，如果我们后面想使用python3.6做为我们的开发环境，则可以使用 `conda create` 命令创建我们的conda虚拟环境。

> conda 创建虚拟环境：
> - conda创建环境命令为：conda create -n your_env_name
> - 创建环境并指定python版本：conda create -n your_env_name python=3.6
> - 创建Pytorch所需环境，输入命令conda create -n torch python=3.6

**step2:** 安装PyCharm
可参考文献[1], 比较简单，略

**step3:** 使用 conda install 命令行安装

pytorch 官方文档推荐的安装命令：
```bash
# 安装最新版
conda install pytorch torchvision cpuonly -c pytorch

# 指定安装版本
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

-c pytorch参数指定了conda获取pytorch的channel, 在此指定为conda自带的pytorch仓库。但是这在国内的下载速度往往很慢，我们可以指定从清华镜像源安装。

```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```
关于使用conda从清华镜像源安装第三方软件包的相关问题可参考文献[2].

文献[2]中采用的是将镜像源直接添加到conda的配置文件中：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ 
conda config --set show_channel_urls yes
```
这些可以在 ~/.condarc文件中修改，先后顺序表示优先级（隐藏文件查看可用 ls -a）。（参考文献[3]）

我们可以点进去上述连接查看，发现不同通道下的第三方包更新不太一样，笔者有的时候也会使用科学上网下载。对于我个人来说，我是在使用`conda install`命令的时候指定镜像源通道，而不是将这些镜像源通道写到conda配置文件。

**我的一般安装思路如下：** \
首先使用镜像源安装，若成功最好 (**此时要关闭科学上网**)；\
若不成功则使用科学上网下载 (**这时候不要使用镜像源**)；\
如果还是下载慢，可以尝试到清华tuna镜像源目录下去找找看哪个链接下有我们需要的软件包吧。


```bash
# 从anaconda官方下载：
conda install -c menpo opencv3

# 从清华镜像源下载：
conda install opencv -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```





# 参考文献
[1] [Anaconda python3.6及pycharm安装和简单使用--Windows](https://sevenold.github.io/2018/11/anaconda-windows/) \
[2] [conda安装tensorflow和pytorch](https://zhuanlan.zhihu.com/p/52498335)
[3] [conda安装tensorflow和conda常用命令](https://blog.csdn.net/YuzuruHanyu/article/details/86186549)