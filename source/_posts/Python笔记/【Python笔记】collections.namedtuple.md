---
title: 【Python笔记】collections.namedtuple
date: 2019-07-18
tags:
categories: ["Python笔记"]
mathjax: true
---
<!-- more -->

## 使用案例1
```python
import collections  

if __name__ == '__main__':  
    TPoint = collections.namedtuple('TPoint', ['x', 'y'])  # 定义一个TPoint class类型，而且带有属性x,y  
  p = TPoint(x=10, y=10)  # 创建TPoint对象  
  print(p.x, p.y)  # 通过类成员变量访问  
  print(p[0], p[1])  # 通过元祖索引访问  
  for v in p:  # 通过for循环迭代访问  
  print(v)
```

## 使用案例2
```python
import collections  
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):  
    """  
    :param
    """  

if __name__ == '__main__':  
    b = Block('block1', max, [{'stride': 2, 'depth_bottleneck': 4, 'depth': 4, 'rate': 1}])  
  
    print('debug')
```
