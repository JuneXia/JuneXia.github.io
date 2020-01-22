---
title: 【开发环境与工具】markdown语法笔记
date: 2017-03-09
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

## 数学符号字体
斜体加粗 A：
$\boldsymbol{A}$

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


https://blog.csdn.net/thither_shore/article/details/52260742


https://blog.csdn.net/deepinC/article/details/81103326
