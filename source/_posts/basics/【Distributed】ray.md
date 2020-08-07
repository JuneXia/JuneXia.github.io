---
title: 
date: 2020-8-7
tags:
categories: ["basics"]
mathjax: true
---

&emsp; Spark 、Flink、Hadoop 是大家比较熟悉的分布式执行引擎，高效可控，缺点是学习曲线陡峭，入门需要懂相当多的基础知识。

有一个比较新的分布式执行引擎 Ray （基于 Python）大大简化了这个过程。

官方开源地址：https://github.com/ray-project/ray

<!--more-->

直接看代码示例吧 [1]。

```python
import time
import ray


def func1():
    time.sleep(1)
    return 1


@ray.remote
def func2():
    time.sleep(1)
    return 1


if __name__ == '__main__':
    t1 = time.time()
    rslt = [func1() for i in range(4)]
    print(rslt, 'execute time: ', time.time() - t1)

# output:
[1, 1, 1, 1] execute time:  4.004237651824951




if __name__ == '__main__':
    ray.init()

    t1 = time.time()
    a = [func2.remote() for i in range(4)]
    rslt = ray.get(a)  # ray.get 会阻塞等待子进程返回
    print(rslt, 'execute time: ', time.time() - t1)

# output: 这里只开了4个进程，如果开更多的进程，则加速效果更显著
[1, 1, 1, 1] execute time:  2.698688268661499
```



# 参考文献
[1] [一行代码变分布式 Ray（Python）](https://zhuanlan.zhihu.com/p/41875076) \
[2] [python分布式多进程框架 Ray](https://blog.csdn.net/luanpeng825485697/article/details/88242020)

