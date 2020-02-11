---
title: 【开发环境与工具 blog】hexo+github搭建博客
date: 2019-08-17
tags: 
categories: ["开发环境与工具"]
mathjax: true
---

本文主讲使用 hexo+github 搭建博客。

<!-- more -->

## 使用 Node installer 安装 Node.js（不推荐此法）
关于Node.js
> Node.js® is a JavaScript runtime built on Chrome's V8 JavaScript engine.

[Node.js下载地址](https://nodejs.org/zh-cn/download/)

本方法安装指导参考[这里](https://github.com/nodejs/help/wiki/Installation)


**安装步骤如下：**
1. Unzip the binary archive to any directory you wanna install Node, I use /usr/local/lib/nodejs

```bash
 sudo mkdir -p /usr/local/lib/nodejs
 sudo tar -xJvf node-v12.14.1-linux-x64.tar.xz -C /usr/local/lib/nodejs 
```

2. Set the environment variable ~/.profile, add below to the end

```bash
# Nodejs
VERSION=v12.14.1
DISTRO=linux-x64
export PATH=/usr/local/lib/nodejs/nodejs/node-v12.14.1-linux-x64/bin:$PATH
```

3. Refresh profile

```bash
. ~/.profile
```

4. Test installation using

```bash
$ node -v
$ npm version
$ npx -v
```

the normal output is:

➜  node -v \
v12.14.1

➜  npm version \
{ npm: '6.4.1',
 ares: '1.15.0',
 cldr: '33.1',
 http_parser: '2.8.0',
 icu: '62.1',
 modules: '64',
 napi: '3',
 nghttp2: '1.34.0',
 node: '10.15.1',
 openssl: '1.1.0j',
 tz: '2018e',
 unicode: '11.0',
 uv: '1.23.2',
 v8: '6.8.275.32-node.12',
 zlib: '1.2.11' }


5. Using sudo to symlink node, npm, and npx into /usr/bin/:
```bash
sudo ln -s /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/node /usr/bin/node

sudo ln -s /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/npm /usr/bin/npm

sudo ln -s /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/npx /usr/bin/npx
```

&emsp; 笔者实测按照上述方法安装node.js和npm后，后面在安装hexo时会出现EACCES权限错误，后遵循 [由 npmjs 发布的指导](https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally) 修复该问题。强烈建议 不要 使用 root、sudo 等方法覆盖权限。
笔者这里最后是通过nvm来安装node.js的。

## 使用 Node version manager(nvm) 安装 Node.js
&emsp; [Node Version Manager(nvm)](https://github.com/nvm-sh/nvm) 是一个开源的node.js多版本管理Bash工具，类似于Python中的pyenv工具，用于在Bash环境中随意切换已安装的node版本

1. 安装nvm
```bash
# 我使用wget安装
$ curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.11/install.sh | bash

or

$ wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.33.11/install.sh | bash
```

2. 使用 nvm 安装 Node.js
```bash
$ nvm install node  # 不推荐
```
> 不推荐，nvm安装的最新版可能并不是Node官方推荐的稳定版本，建议参考[官方推荐版本](https://github.com/nvm-sh/nvm)后进行版本指定安装：

```bash
$ nvm ls-remote  # 查看可用的node.js版本

$ nvm install 12.14.1
```

## 安装Hexo
```bash
$ npm install -g hexo-cli

# 为了便于发布到GitHub上，建议同时安装hexo-deployer-git
$ npm install hexo-deployer-git --save
```

## 建立博客
&emsp; 这里分两部分来讲博客的建立，一个是从头建立，另一个是从已有博客建立。

### 从头建立博客
```bash
$ hexo init myblog
$ npm install

# 常用命令简写
$ hexo n "title_name" == hexo new "title_name" #新建文章
$ hexo g == hexo generate #生成
$ hexo s == hexo server #启动服务预览
$ hexo d == hexo deploy #部署

$ hexo server #Hexo会监视文件变动并自动更新，无须重启服务器
$ hexo server -s #静态模式
$ hexo server -p 5000 #更改端口
$ hexo server -i 192.168.1.1 #自定义 IP
$ hexo clean #清除缓存，若是网页正常情况下可以忽略这条命令
```

**hexo常用命令积累**
```bash
$ hexo s --debug
```

### 从已有博客建立博客
&emsp; 假设我们现在已经有了博客了，但后期如果我们重装系统了，或者想在另一台电脑上搭建博客环境，这时候我们需要重新搭建我们之前的博客内容.我们需要拉取我们的hexo分支,因为hexo分支是源文件,master可以不用拉取

```bash
# 克隆分支到本地
git clone -b hexo https://github.com/用户名/仓库名.git
# 进入博客文件夹
cd youname.github.io

# 安装依赖，记得此时不需要hexo init这条指令
npm install hexo
npm instal
npm install hexo-deployer-git
```

发布预览
```bash
hexo clean && hexo g && hexo d
```

## 更换Hexo主题
1. 下载主题到themes目录
&emsp; 我这里使用的是next主题，参见 [next主题官方指导](https://theme-next.iissnan.com/getting-started.html)
git clone 会得到最新的next版本，但不一定是稳定版本，我是到 [发布页面](https://github.com/iissnan/hexo-theme-next/releases)下载的最新的稳定版源代码，并解压到themes目录下，并更改文件名为next

2. 将_config.yml配置文件设置themes主题为next
```bash
theme: next
```

3. 本地发布预览
```bash
$ hexo s --debug
```
