---
title: 
date: 2020-9-17
tags:
categories: ["Python笔记"]
mathjax: true
---

<!--more-->


```python
"""
References: [Python 优雅的使用参数 - 可变参数（*args & **kwargs)](https://n3xtchen.github.io/n3xtchen/python/2014/08/08/python-args-and-kwargs)
"""


# 形式1：行参前有一个*，表示args传入的是一个tuple
def func1(*args):
    print(args)
    for arg in args:
        for element in arg:
            print(element)
            # yield element


# 形式1：行参前有两个**，表示kwargs传入的是一个dict
def func2(**kwargs):
    print(kwargs)
    for k, v in kwargs.items():
        print(k, v)


if __name__ == '__main__':
    func1([1, 2])
    func1(1, 2)
    func1((1, 2), 's')
    func2(a=1, b='s')
```


可变参数的应用：

```python
def fsum(a, b, c):
    sum = a + b + c
    return sum


# 写法1:
def method1(callback, *args):
    print(args)
    return callback(*args)

# 写法2:
def method2(callback, **kwargs):
    print(kwargs)
    return callback(**kwargs)


if __name__ == '__main__':
    print('method 1 result: ', method1(fsum, 1, 2, 3))
    print('method 2 result: ', method2(fsum, a=1, b=2, c=3))
```
