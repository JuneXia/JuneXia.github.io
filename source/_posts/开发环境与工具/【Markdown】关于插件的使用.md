---
title: 
date: 2020-02-12
tags:
categories: ["开发环境与工具"]
mathjax: true
---
# 为了解决Graph预览问题
在VS-Code中使用 Markdown All in One 无法支持 mermaid 图表的预览，一顿搜索之后发现大家推荐安装 Markdown Preview Enhanced 插件来解决该问题。
<!-- more -->

VS-Code下安装插件简单，Markdown Preview Enhanced 安装完成后，重启 VS-Code，然后找一个.md文件，在里面右键选择 “Markdown Preview Enhanced” 便可预览

**修改 Markdown Preview Enhanced 主题**
打开 File > Preferences > Settings，搜索“Markdown Preview Enhanced:Theme”按个人需求更改自己的theme配置，我这里preview选择的是“none.css”，记住配置作用范围只能选择Workspace（User作用于更改不了）

**Markdown Preview Enhanced 不支持Latex自动补全**
喔靠，为了预览个Graph，把Letax自动补全给舍弃？不可能的。
暂且的做法就是：“Markdown All in One”和“Markdown Preview Enhanced”都Enable，一般情况下还是使用VS-Code自带的预览功能吧，如果要预览Graph就临时用“Markdown Preview Enhanced”看下吧，毕竟最后是要发布到hexo的，只要hexo都支持显示就ok了。


