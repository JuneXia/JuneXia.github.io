---
title: 
date: 2015-05-19
tags:
categories: ["Math"]
mathjax: true
---

## 映射

### 映射的概念

定义见课本

简说：对于给定的 $x \in D_f$，按照法则 $f$，有唯一确定的 $y \in R_f$


**TIPS**: \
设有映射 $f: X \rightarrow Y$， \
则我们约定(无需证明)：映射 $f$ 的定义域$D_f = X$，映射 $f$ 的值域$R_f \subset Y$，不一定有 $R_f = Y$；\
如果 $R_f = Y$（**即** $Y$ 中任一元素 $y$ 都是 $X$ 中某元素的像），则称 $f$ 为 $X$ 到 $Y$ 的**满射**；\
若对 $X$ 中任意两个不同元素 $x_1 \not ={x_2}$，它们的像 $f(x_1) \not ={f(x_2)}$（单调），$\leftrightarrows$ 则称 $f$ 为 $X$ 到 $Y$ 的**单射**；（这句话其实就是说：单调 $\rightarrow$ 单射）\
若映射 $f$ 既是单射又是满射，则称 $f$ 为**一一映射**(或**双射**)


### 逆映射

定义见课本

简说：设 $f$ 为单射，对于给定的 $y \in R_f$，按照法则 $f$，有唯一确定的 $x \in D_f$. 于是我们可以定义一个从 $R_f$ 到 $D_f$ 的新映射 $f^{-1}$

**TIPS**:\
只有单射才存在逆映射.


### 复合映射
设有两个映射
$$
g: X \rightarrow Y_1, \qquad f: Y_2 \rightarrow Z
$$
映射 $g$ 和 $f$ 构成复合映射的条件是：$R_g \subset D_f$


## 函数

### 函数的概念
略

### 函数的几种特性

#### 有界性
设$X \subset D$，对于任一 $x \in X$，如果存在数 $K_1$ 使得
$$
f(x) \leq K_1
$$
成立，则称函数$f(x)$在$X$上有**上界**.

另有**下界、有界**的定义类似。

#### 单调性


#### 奇偶性


#### 周期性


### 反函数与复合函数

反函数为逆映射的特例，复合函数为复合映射的特例。其实都与逆映射和复合映射类似，这里就不再赘述了。

需要注意的是：相对于反函数 $y = f^{-1}(x)$ 来说，原来的函数 $y = f(x)$ 称为**直接函数**，如果把直接函数和反函数的图行画在同一个坐标平面上，则这两个图形是关于 $y=x$ 对称的。

### 函数的运算

### 初等函数

关于三角函数以及反三角函数的曲线图可参见文献[1]

-------------------
### 双曲函数和反双曲函数

双曲余弦函数 $y=\text{ch} \ x$ 如下图15所示：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/math/highmath1-1-1functions_and_limits1.jpg" width = 60% height = 60% />
</div>

课本推导的$y=\text{ch} \ x$当$x \geq 0$时它的反函数$y=\text{ch} \ x$的图像如下：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/math/highmath1-1-1functions_and_limits2.jpg" width = 60% height = 60% />
</div>

**TIPS**: 
1. 课本在讨论 $y=\text{ch} \ x$ 的反函数时，只讨论了当 $x \geq 0$ 时的情况，那么当 $x \leq 0$ 时的情况又是怎么样的呢？
2. $y=\text{ch} \ x$ 和其反函数 $y=\text{arch} \ x$ 不应该是关于 $y=x$ 对称的吗？

我的理解：
反函数是逆映射的一个特例

逆映射$f^{-1}$存在的条件是$f$是单射，而根据单射的定义，单射其实就是单调。\
观察$y=\text{ch} \ x$，其在定义域上显然不是单调的，所以它在定义域上没有反函数。\
但它在$(-\infty, 0]$和$[0, + \infty)$这两个子区间上是单调的，所以$y=\text{ch} \ x$的反函数可以分区间来讨论，课本已经讨论了$[0, + \infty)$这个子区间，而在这个子区间上，$y=\text{ch} \ x$和其反函数$y=\text{arch} \ x$很明显是对称的。 \
按照课本类似的推导方法，我们也可以推导出$y=\text{ch} \ x$在$(-\infty, 0]$区间的反函数，实际上就是图1-18中$y=\text{arch} \ x$关于x轴对称的部分。\
这时候$y=\text{arch} \ x$当x取定义域$x \geq 1$中的任一个数，都有两个y值与之对应，这不符合我们目前单值函数的范畴了。\
所以，对于$y=\text{ch} \ x$，如果要考虑它在整个定义域上的反函数，则它是没有反函数的，但是如果考虑它在$(-\infty, 0]$和$[0, + \infty)$这个任意一个子区间上的反函数的话，那么它是有反函数的。



# 参考文献
[1] [三角函数与反三角函数(图像)](http://math001.com/inverse_trigonometric_functions/)


----------------------

# 数列极限的定义

课本进度：p23 定理1


V研客：

极限的概念

极限的性质

极限存在准则

无穷小

无穷大



> 常考题型
> 1. 极限的概念性质及存在准则
> 2. 求极限
> 3. 无穷小量阶的比较



# 极限的概念

## 数列的极限
**定义1**： 

$$
\begin{aligned}
  & \lim_{x \to +\infty} x_n = a: \\
  & \forall \epsilon > 0, \text{while } n > N, \text{constant has } \vert x_n - a \vert < \epsilon.
\end{aligned}
$$

【注】：\
1. $\epsilon$ 与 N 的作用：\
**任意的** $\epsilon$ 是为了刻画 $x_n$ 与 $a$ 的接近程度；\
**N** 是为了刻画下标n无限增大的过程。

2. 几何意义 \
n > N 后的无穷多项都落在了以 a 中心的领域$(a - \epsilon, a + \epsilon)$内（当然，n < N 的时的$x_n$也有可能落在这个领域）;
落在以 a 中心的领域中有无限多项，而落在其他地方的则是有限项；

3. 数列有没有极限，跟前有限项没有关系，而是跟大于N后的无限项有关系;

4. 如果$x_n$极限存在，则他的任意子数列的极限也存在且和原极限相等，但反之未必。例如子数列$x_{2k-1}$和$x_{2k}$的极限存在但不相等，这时候就不同通过子数列极限存在导出原数列极限也存在的结果了。\
$$
\lim_{n \to \infty} x_n = a \Leftrightarrow \lim_{k \to \infty} x_{2k-1} = \lim_{k \to \infty} x_{2k} = a.
$$

