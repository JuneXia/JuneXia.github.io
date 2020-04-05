---
title: 【开发环境与工具】git使用
date: 2017-03-11
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

# git 版本管理

**本地更新而远程未更新，选择提交本地文件**
```bash
// 查看当前本地文件和远程文件的变化状态
git status

// 选择你想要提交的文件修改或添加
git add file1 file2 file3

// 此时再用git status命令就看不到刚刚已经add过的文件状态了

git commit -m '提交说明信息'

git push
```


**比较本地仓库与远程仓库的区别**
```bash
git fetch orgin

git diff master origin/master
或者
git diff master origin/其他分支
```


# git 杂记
## 解决git status不能显示中文
- 现象：
status查看有改动但未提交的文件时总只显示数字串，显示不出中文文件名，非常不方便。如下图：


- 原因：
在默认设置下，中文文件名在工作区状态输出，中文名不能正确显示，而是显示为八进制的字符编码。

- 解决办法：
将git 配置文件 core.quotepath项设置为false；\
quotepath表示引用路径 \
加上--global表示全局配置 \

git bash 终端输入命令：
```bash
git config --global core.quotepath false
```

## 解决 git bash 终端显示中文乱码
参考文献[1]


# 参考文献
[1] [git status 显示中文和解决中文乱码](https://blog.csdn.net/u012145252/article/details/81775362)
