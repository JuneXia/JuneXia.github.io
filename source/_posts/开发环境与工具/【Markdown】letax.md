---
title: 
date: 2018-05-12
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

# 数学字体
| 需求 | latex语法 | 显示效果 | 备注 |
| ------ | ------ | ------ | ------ |
| 正体加粗 | \bold{A} | $\bold{A}$ | 该方法在vscode中可显示，但发布到hexo后不能正确显示 |
| 正体加粗 | \textbf{x} | $\textbf{x}$ | 支持对数字英文字母加粗，但不支持对希腊字母加粗 |
| 正体加粗 | \mathbf{x} | $\mathbf{x}$ | 支持对数字英文字母加粗，但不支持对希腊字母加粗 |
| 斜体加粗 | \boldsymbol{A} | $\boldsymbol{A}$ | 将正体变为斜体加粗，支持对希腊字母加粗 |


# 数学符号
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
| 一阶偏导 | \frac{\partial f}{\partial x} | $\frac{\partial f}{\partial x}$ | |
| n阶偏导 | \frac{\partial^{n} f}{\partial x^{n}} | $\frac{\partial^{n} f}{\partial x^{n}}$ | |
| 一阶导数 | \frac{\mathrm{d} y }{\mathrm{d} x} | $\frac{\mathrm{d} y }{\mathrm{d} x}$ | |
| n阶导数 | \frac{\mathrm{d}^{n} y }{\mathrm{d} x^{n}} | $\frac{\mathrm{d}^{n} y }{\mathrm{d} x^{n}}$ | |
| 点形式的求导符号 | \frac{ \dot y }{ \dot x } | $\frac{ \dot y }{ \dot x }$ | 一个点 |
| 点形式的求导符号 | \frac{ \ddot y }{ \ddot x } | $\frac{ \ddot y }{ \ddot x }$ | 两个点 |
| 波浪号 | \sim | $\sim$ |  |
| 约等于 | \approx | $\approx$ |  |
| 斜体 | log (x) | $log (x)$ |  |
| 正体 | \text{log } (x) | $\text{log} (x)$ |  |
| 正体 | \text{log } x | $\text{log } x$ | text中可以写空格 |
| 花体 | \mathscr{L} | $\mathscr{L}$ |  |
| 花体 | \mathcal{L} | $\mathcal{L}$ |  |
| 空心字母 | \mathbb{R} | $\mathbb{R}$ | |	
| 等号下面 | \underset{\text{heated}}{=} | $\underset{\text{heated}}{=}$ | 其实可以是任意符号下面 |
| 等号上面 | \overset{\text{def}}{=} | $\overset{\text{def}}{=}$ | 其实可以是任意符号上面 |
| 等号上面三角形 | \triangleq | $\triangleq$ | 表示“定义为” |
| max下面 | \max \limits_{f} | $\max \limits_{f}$ |  |
| min下面 | \min \limits_{f} | $\min \limits_{f}$ |  |
| 取整函数（Floor function） | \lfloor a+b \rfloor | $\lfloor a+b \rfloor$ |  |
| 取整函数（Floor function） | \left \lfloor a+b \right \rfloor | $\left \lfloor a+b \right \rfloor$ | 似乎这种较上面更繁琐一些 |
| 取顶函数（Ceiling function） | \lceil a+b \rceil | $\lceil a+b \rceil$ |  |
|  | \odot | $\odot$ |  |
|  | \oplus | $\oplus$ |  |


$f_{\limits_{f}}$



## 数学公式中的空格
| | | | |
| --- | --- | --- | --- |
| 两个quad空格 | a \qquad b | $a \qquad b$ | 两个m的宽度 |
| quad空格 | a \quad b | $a \quad b$ | 一个m的宽度 |
| 大空格 | a\ b | $a\ b$ | 1/3m宽度 |
| 中等空格 | a\\;b | $a\;b$ | 2/7m宽度 |
| 小空格 | a\\,b | $a\,b$ | 1/6m宽度 |
| 没有空格 | ab | $ab$ | |		 
| 紧贴 | a\\!b | $a\!b$ | 缩进1/6m宽度 (**注意**: 这种写法在vscode中没问题，但发布到hexo后显示不正确。再hexo中正确的做法是: 1. 对于换行公式可以使用 \\! 紧贴; 2. 而对于行内公式要使用 \\\! 紧贴) |
| 符号尺寸 | \big( \bigg( \Big( \Bigg( | $\big( \bigg( \Big( \Bigg($ | 这里是以圆括号为例，实际上对于任意符号都可 |



\quad、1em、em、m代表当前字体下接近字符‘M’的宽度。


# 方程组

**注意事项**：
1. 以\begin和\end包裹的公式，如果要加 tag 标签，则应该加在 \end{...}之后
2. 在公式中显示星号(\*)在vscode中直接输入就可以了，但是在发布到hexo则不行。\
   如果要发布到hexo，则行内公式如果要显示星号(\*)，则需要加反斜杠(\\)对其进行转义；而换行公式则不必如此


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

# 异常问题处理
## 异常1： hexo无法发布双大括号的问题
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
[CSDN-markdown 之 LaTeX 特殊公式格式笔记](https://blog.csdn.net/thither_shore/article/details/52260742)

[markdown数学公式符号记录](https://blog.csdn.net/deepinC/article/details/81103326)

[常用数学符号的 LaTeX 表示方法](http://www.mohu.org/info/symbols/symbols.htm)

