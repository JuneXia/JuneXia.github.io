---
title: 【Python笔记】numpy排序
date: 2017-08-20
tags:
categories: ["Python笔记"]
mathjax: true
---

numpy排序有好几个，如sort，sorted，argpartition。笔者用到哪个就介绍哪个吧
<!-- more -->

# np.argpartition

## 找出数组中的第n小或第n大值的下标
```python
# 输出arr中第0小值(即最小值)的下标
np.argpartition(arr, 0)[0]

Out[19]:
9

# 输出arr中第1小值的下标
np.argpartition(arr, 1)[1]

Out[20]:
8

# 输出arr中第len(arr)-1小值(即最大值)的下标
np.argpartition(arr, len(arr) - 1)[len(arr) - 1]

Out[24]:
0

# 输出arr中第1大的值的下标
np.argpartition(arr, -1)[-1]

Out[25]:
0

# 输出arr中第2大的值的下标
np.argpartition(arr, -2)[-2]

Out[26]:
1

# 同时找到arr中第2和第4小值的下标，然后输出第2小值的下标
np.argpartition(arr, [2, 4])[2]

Out[27]:
7

# 同时找到arr中第2和第4小值的下标，然后输出第4小值的下标
np.argpartition(arr, [2, 4])[4]

Out[28]:
5
```
或许有人会问，为什么不对数组arr做个排序，然后再输出呢。
这是因为np.argpartition比“先排序再输出”这中做法效率更高，np.argpartition并没有对数组中所有的数都做了排序，下面代码见分晓。

## 取出数组中前n小数值的下标
```python
import numpy as np
arr = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

arr_part = np.argpartition(arr, 4)
arr_part

Out[14]:
array([7, 8, 9, 6, 5, 4, 1, 3, 2, 0])

# 取出第4小数的下标 (下标从0开始)
arr_part[4]

Out[17]:
5

# 有人会问 arr_part[3]是第3小数的下标吗？答案是否定的
# 仔细看看arr_part中的数值分布，我们会发现下标4的左边都是比arr_part[4]小的数的下标，而右边都是比arr_part[4]大的数的下标。
# 但左右两边这些下标并不一定都是按照数值从小到大排列的。

# 但是我们可以取出前4小数值的下标
arr_part[:4]

Out[18]:
array([7, 8, 9, 6])
```

相应的，也可以输出前n大的数值的下标。


# np.argsort
```python
# 先定义一个测试array
arr = np.array([[5, 3, 4],
                [4, 5, 6],
                [9, 8, 9],
                [7, 6, 1],
                [4, 5, 7]])
```

## 指定按某一列排序
```python
# 先得到所有按列排序的下标，axis=0,表示按列排序
index = np.argsort(arr, axis=0)
index
Out[21]: 
array([[1, 0, 3],
       [4, 1, 0],
       [0, 4, 1],
       [3, 3, 4],
       [2, 2, 2]])

# 指定按第0列排序
arr[index[:, 0]]
Out[30]: 
array([[4, 5, 6],
       [4, 5, 7],
       [5, 3, 4],
       [7, 6, 1],
       [9, 8, 9]])

# 指定按第1列排序
arr[index[:, 1]]
Out[31]: 
array([[5, 3, 4],
       [4, 5, 6],
       [4, 5, 7],
       [7, 6, 1],
       [9, 8, 9]])
```


## 指定按某一行排序
按行排序的道理和上面类似。
```python
# 先得到所有按行排序的下标
index = np.argsort(arr, axis=1)
index
Out[19]: 
array([[1, 2, 0],
       [0, 1, 2],
       [1, 0, 2],
       [2, 1, 0],
       [0, 1, 2]])

# 指定按第0行排序
arr[:, index[0, :]]
Out[47]: 
array([[3, 4, 5],
       [5, 6, 4],
       [8, 9, 9],
       [6, 1, 7],
       [5, 7, 4]])

# 指定按第1行排序
arr[:, index[1, :]]
Out[48]: 
array([[5, 3, 4],
       [4, 5, 6],
       [9, 8, 9],
       [7, 6, 1],
       [4, 5, 7]])
```



# 参考文献
[1] [Python库Numpy的argpartition函数浅析](https://blog.csdn.net/weixin_37722024/article/details/64440133)


