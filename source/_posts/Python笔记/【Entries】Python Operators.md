---
title: 
date: 2020-02-01
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述Python的两个操作符\*和\*\*的作用。
<!-- more -->

# 用作运算符
\*\*两个乘号就是乘方，比如2\*\*4, 结果就是2的4次方，结果是16.

一个乘号\*，如果操作数是两个数字，就是这两个数字相乘，如2\*4,结果为8.

\*如果是字符串、列表、元组与一个整数N相乘，返回一个其所有元素重复N次的同类型对象，比如"str"\*3将返回字符串"strstrstr"


# 用作操作符
如果是在函数定义中，则参数前的\*表示的是将调用时的多个参数放入元组中，\*\*表示将调用函数时的关键字参数放入一个字典中。

如定义以下函数：\
```python
def func(*args):
    print(args)
```
当用 func(1, 2, 3) 调用函数时，参数 args 就是元组 (1, 2, 3)

```python
def func(**args):
    print(args)
```
当用 func(a=1, b=2) 调用函数时，参数 args 将会是字典 {'a': 1, 'b': 2}


如果是在函数调用中，\*args 表示将**可迭代对象**扩展为函数的参数列表，而 \*\*args 表示将字典扩展为关键字参数。

例如：
```python
args = (1, 2, 3)
func = (*args)
```
等价于函数调用 func(1, 2, 3)

```python
args = {'a': 1, 'b': 2}
func(**args)
```
等价于函数调用 func(a=1, b=2)


# 参考文献
[1] [python 操作符**与*的用法](https://blog.csdn.net/zhihaoma/article/details/50572854)
