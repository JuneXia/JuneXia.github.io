---
title: 【开发环境与工具】git使用
date: 2017-03-11
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

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


