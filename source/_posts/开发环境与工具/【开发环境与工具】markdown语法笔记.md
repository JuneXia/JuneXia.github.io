---
title: 【开发环境与工具】markdown语法笔记
date: 2017-03-09
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

## 显示图片
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/center-loss1.jpg" width = 60% height = 60% />
</div>
<center>图1 &nbsp;  分类任务和人脸识别任务对Features的要求比较</center>

## 数学字体
| 需求 | latex语法 | 显示效果 | 备注 |
| ------ | ------ | ------ | ------ |
| 加粗 | \bold{A} | $\bold{A}$ ||
| 斜体加粗 | \boldsymbol{A} | $\boldsymbol{A}$ ||

# markdown特殊符号


<br>

## 数学符号
| 需求 | latex语法 | 显示效果 | 备注 |
| ------ | ------ | ------ | ------ |
| 点乘 | a \cdot b | $a \cdot b$ ||
| 叉乘 | a \times b | $a \times b$ ||
| 点除 | a \div b | $a \div b$ ||
| 分数 | \frac {a} {b} | $\frac{a}{b}$ ||
| 绝对值号 | | 或者 \vert | $\vert$ ||
| 范数符号 | \| 或者 \Vert | $\Vert$ ||
| 大于等于号 | \geq | $\geq$ | 后面记得加空格，不然识别出错 |
| 小于等于号 | \leq | $\leq$ | |
| 等于号 | \not = | $\not =$ | |
|  | \land | $\land$ | |
| 指示函数符号 | \mathbb I | $\mathbb I$ | 周志华机器学习书本中出现过 |


## 数学公式中的空格
| | | | |
| --- | --- | --- | --- |
| 两个quad空格 | a \qquad b | $a \qquad b$ | 两个m的宽度 |
| quad空格 | a \quad b | $a \quad b$ | 一个m的宽度 |
| 大空格 | a\ b | $a\ b$ | 1/3m宽度 |
| 中等空格 | a\\;b | $a\;b$ | 2/7m宽度 |
| 小空格 | a\\,b | $a\,b$ | 1/6m宽度 |
| 没有空格 | ab | $ab$ | |		 
| 紧贴 | a\\!b | $a\!b$ | 缩进1/6m宽度 |


\quad、1em、em、m代表当前字体下接近字符‘M’的宽度。


## 方程组
1
$$
\begin{pmatrix}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{pmatrix}
$$

2
$$
\begin{gathered}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{gathered}
$$

3
$$
\begin{aligned}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{aligned}
$$

4
$$
\begin{alignedat}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{alignedat}
$$

5
$$
\begin{bmatrix}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{bmatrix}
$$

6
$$
\begin{Bmatrix}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{Bmatrix}
$$

7
$$
\begin{cases}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{cases}
$$

8
$$
\begin{dcases}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{dcases}
$$

9
$$
\begin{Vmatrix}
    min \ f(x_1, x_2, ..., x_n) \\
s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{Vmatrix}
$$

10
> &表示以等号为对齐标记
$$
\begin{aligned}  
f(x) &= (m+n)^2 \\  
&= m^2+2mn+n^2 \\  
\end{aligned}  
$$

11
> 这里没有等号，要想实现左对齐，可以在左侧直接加一个 &
$$
\begin{aligned}  
& min \ f(x_1, x_2, ..., x_n) \\
& s.t. \ h_k(x_1, x_2, ..., x_n) = 0 ,\quad  (k=1,2,...,l)
\end{aligned}  
$$

12
$
\begin{aligned}
f(x) &= (m+n)^2 \\
&= m^2+2mn+n^2 \\
\end{aligned}
$

## 异常问题处理
### 异常1： hexo无法发布双大括号的问题
```latex
$$
\dfrac{\partial f}{\partial x} + \mu_1 \dfrac{{\rm d}g_1}{{\rm d}x} + \mu_2 \dfrac{{\rm d}g_2}{{\rm d}x}= 0,
$$
```

上述latex在vscode下是正常显示的，但是hexo无法发布

解决办法：双大括号间加入空格即可
```
在{{之间加入空格，}{这种双大括号无碍
```


# 参考文献
https://blog.csdn.net/thither_shore/article/details/52260742


https://blog.csdn.net/deepinC/article/details/81103326

[常用数学符号的 LaTeX 表示方法](http://www.mohu.org/info/symbols/symbols.htm)
