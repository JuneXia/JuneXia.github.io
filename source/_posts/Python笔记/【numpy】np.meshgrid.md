---
title: 
date: 2020-05-01
tags:
categories: ["Python笔记"]
mathjax: true
---

更多内容请参考文献[1]

**示例1：**
```python
np.meshgrid([1, 2], [2,3,4])

# output:
"""
[
    array([[1, 2],
           [1, 2],
           [1, 2]]), 

    array([[2, 2],
           [3, 3],
           [4, 4]])
]
"""
```

以上会生成meshgrid坐标矩阵：
```python
[ (1, 2), (2, 2), (1, 3), (2, 3), (1, 4), (2, 4)]
```

**示例2：**
```python
np.meshgrid([[1, 2], [5, 6]], [2,3,4])

# output: 
"""
[
    array([[1, 2, 5, 6],
           [1, 2, 5, 6],
           [1, 2, 5, 6]]), 

    array([[2, 2, 2, 2],
           [3, 3, 3, 3],
           [4, 4, 4, 4]])
]
"""
```
如果 `np.meshgrid(X, Y)` 中的 X 或者 Y 是二维矩阵的话，则 np.meshgrid 内部会先将它们做 flatten 操作，然后再进行 meshgrid 操作。


# 参考文献
[1] [Python-Numpy模块Meshgrid函数](https://zhuanlan.zhihu.com/p/33579211)