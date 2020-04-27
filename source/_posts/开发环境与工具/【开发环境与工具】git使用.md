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

## git撤销已经push的提交

参考文献[2]

**step1**: 
使用 git log 命令查看提交记录，如：
`xj@win$ git log`

```
commit 6c251cca5f6cdb1aa8850737009f132894deab5e
Author: 名字 <name@a.com.cn>
Date: Thu Dec 13 14:29:21 2018 +0800
 
mobilenet
 
commit 43dc0de914173a1a8793a7eac31dbb26057bbee4
Author: 名字 <name@a.com.cn>
Date: Thu Dec 13 13:54:32 2018 +0800
 
yolov1
```

**step2**： 
我们要撤销“mobilenet”这个提交，即回退到“yolov1”这个提交的版本，也就是回退到commit为“43dc0de914173a1a8793a7eac31dbb26057bbee4”的版本。

使用命令：git reset --soft 43dc0de914173a1a8793a7eac31dbb26057bbee4

最后再次使用git log查看是否成功撤销了本地提交。

> 其中： \
> 参数soft指的是：保留当前工作区，以便重新提交 。
> 还可以选择参数hard，会撤销相应工作区的修改，一定要谨慎使用。

**step3**: 最后，使用git push origin master --force 强制推送版本

其中：master表示远端分支。

如果不加--force会报错，因为版本低于远端，无法直接提交。




# 参考文献
[1] [git status 显示中文和解决中文乱码](https://blog.csdn.net/u012145252/article/details/81775362)
[2] [git撤销已经push的提交](https://blog.csdn.net/wodeshouji6/article/details/84988617)

