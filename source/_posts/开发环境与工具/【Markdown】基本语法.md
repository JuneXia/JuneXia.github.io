---
title: 
date: 2017-03-09
tags:
categories: ["开发环境与工具"]
mathjax: true
---
本篇主要记录一些Markdown基本语法方面的问题，个人笔记，本人已经熟知的语法，这里就不贴了。
<!-- more -->

# 图片显示问题
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/center-loss1.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp;  分类任务和人脸识别任务对Features的要求比较</center>

<br><br>


<html>
    <table style="margin-left: auto; margin-right: auto;">
        <tr>
            <td>
                <!--左侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomaffine_shear1.jpg" width = 50% height = 50% />
                </div>
            </td>
            <center>(a) &nbsp; 在y轴错切</center>
            <td>
                <!--右侧内容-->
                <div align=center>
                <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/transforms_randomaffine_shear2.jpg" width = 50% height = 50% />
                </div>
            </td>
            <center>(b) &nbsp; 在x轴错切</center>
        </tr>
    </table>
    <center>图2 &nbsp; 仿射变换之错切(左边为在y轴错切, 右边为在x轴错切)</center>
</html>


# markdown功能相关
## 注释
我们在写作过程中，经常需要给一句话做一些注释，入出处或者解释之类的[^1]，这时候就需要用到markdown的注释功能了。(实测好像不行)

[^1]: 具体注释内容在这里


<br>


# graph

```mermaid
graph LR
A[RCNN] -- SPPNet --> B[FastRCNN]
B -- RPN --> C[FasterRCNN]
```