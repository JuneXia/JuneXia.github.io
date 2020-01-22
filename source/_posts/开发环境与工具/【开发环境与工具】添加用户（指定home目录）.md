---
title: 【开发环境与工具】添加用户（指定home目录）
date: 2019-09-27
tags:
categories: ["开发环境与工具"]
mathjax: true
---

公司就一台服务器，大家都要用，固态硬盘通常比较小，我们公司的固态硬盘只有512G，倒是有一块10T的机械硬盘，能否将每个人的home目录默认就是放在这个硬盘上的呢？答案是可以的。
<!-- more -->

假设 /disk1是已经挂载好的一块硬盘，（关于如何挂载机械硬盘请参见我的另一篇博文）
step1: 在/disk1目录下创建一个home目录
sudo mkdir -p /disk1/home

step2: 添加用户(指定home目录)，假设我们要创建一个用户名为lucy的用户
sudo useradd -d /disk1/home/lucy -m -s /bin/bash lucy

step3: 为新添加的用户设置密码
sudo passwd lucy

step4: 赋予该用户权限
sudo chmod +w /etc/sudoers
sudo vi /etc/sudoers
![enter image description here](https://lh3.googleusercontent.com/-QBN5e4eqZBE/XNFA5YghutI/AAAAAAAAAOE/Fscwh3px8AwAR4N-vmOM0hhBBWmf5e-HQCLcBGAs/s0/%25E8%25B5%258B%25E4%25BA%2588%25E6%258C%2587%25E5%25AE%259A%25E7%2594%25A8%25E6%2588%25B7sudoers%25E6%259D%2583%25E9%2599%2590.png "赋予指定用户sudoers权限.png")

sudo chmod -w /etc/sudoers   (一定要记得去掉/etc/sudoers的写权限)

step5: ssh登录
通过以上步骤便可使用刚创建的用户身份通过ssh命令登录了。
