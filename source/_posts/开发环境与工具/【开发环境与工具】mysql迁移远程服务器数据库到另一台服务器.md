---
title: 【开发环境与工具】mysql迁移远程服务器数据库到另一台服务器
date: 2019-11-25
tags:
categories: ["开发环境与工具"]
mathjax: true
---

问题描述：需要将远程服务器(云服务器)上的数据库迁移到另一台服务器上
<!-- more -->

解决办法：使用MySQL Workbench先将云服务器数据库导出到本地，然后再将本地的数据库文件导入到另一台服务器上。

1. 导出数据库到本地
参考文献[1]


2. 导入数据库文件到服务器

![](https://github.com/JuneXia/JuneXia.github.io/raw/master/images/envtool/MySQL-Workbench%E5%AF%BC%E5%85%A5%E6%95%B0%E6%8D%AE%E5%BA%93%E5%88%B0%E6%9C%8D%E5%8A%A1%E5%99%A8.jpg)


[1] [# MySQL Workbench导出数据库](https://blog.csdn.net/konglongaa/article/details/54923248)
