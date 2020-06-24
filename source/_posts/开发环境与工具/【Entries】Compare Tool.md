---
title: 
date: 2020-06-23
tags:
categories: ["开发环境与工具"]
mathjax: true
---

转自文献[1]

<!-- more -->

一、原理
Beyond Compare每次启动后会先检查注册信息，试用期到期后就不能继续使用。解决方法是在启动前，先删除注册信息，然后再启动，这样就可以永久免费试用了。

二、下载
首先下载Beyond Compare最新版本，链接如下：[下载地址](https://www.scootersoftware.com/download.php)

三、安装
下载完成后，直接安装。

四、创建BCompare文件
1.进入Mac应用程序目录下，找到刚刚安装好的Beyond Compare，路径如下/Applications/Beyond Compare.app/Contents/MacOS。（没有的话可以将Beyond Compare.app移动到对应目录）

2.修改启动程序文件BCompare为BCompare.real。

3.在当前目录下新建一个文件BCompare，文件内容如下：
```bash
#!/bin/bash
 
rm "/Users/$(whoami)/Library/Application Support/Beyond Compare/registry.dat"

"`dirname "$0"`"/BCompare.real $@
```

4.保存BCompare文件。

5.修改文件的权限：
```bash
chmod a+x /Applications/Beyond\ Compare.app/Contents/MacOS/BCompare
```

以上步骤完成后，打开Beyond Compare就可以正常使用了。


[1] [Mac配置Beyond Compare永久使用](https://blog.csdn.net/double_happiness/article/details/88551249)

