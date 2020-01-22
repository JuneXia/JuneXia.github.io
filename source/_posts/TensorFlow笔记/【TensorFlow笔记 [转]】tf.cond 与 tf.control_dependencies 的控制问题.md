---
title: 
date: 2017-09-16
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---

本文转自：[tf.cond 与 tf.control_dependencies 的控制问题](http://yanjoy.win/2017/04/18/tfcond/)
<!-- more -->

# 问题引入
在搜索`tf.cond`的使用方法时，找到了这样的一个问题：

运行下面的一段tensorflow代码：  
```python
pred = tf.constant(True)  
x = tf.Variable([1])  
assign_x_2 = tf.assign(x, [2])  
def update_x_2():  
 with tf.control_dependencies([assign_x_2]):  
 return tf.identity(x)  
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))  
with tf.Session() as session:  
 session.run(tf.initialize_all_variables())  
 print(y.eval())  
```

从代码上看，`tf.cond`经过判断`pred`的值对`x`进行更新。但实际上无论在pred = Ture 还是 False，输出的结果都是2，都是`pred = tf.constant(True)`的情况。

[Confused by the behavior of  `tf.cond`](http://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond)

这是怎么回事呢？



# 顺序执行
先不进行解释，有人在回复中给出了一个可以正确运行的代码，看一下有什么区别：  
```python
pred = tf.placeholder(tf.bool, shape=[])  
x = tf.Variable([1])  
def update_x_2():  
 with tf.control_dependencies([tf.assign(x, [2])]):  
 return tf.identity(x)  
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))  
with tf.Session() as session:  
 session.run(tf.initialize_all_variables())  
 print(y.eval(feed_dict={pred: False}))  # ==> [1]  
 print(y.eval(feed_dict={pred: True}))   # ==> [2]  
```

区别也不大，只是把`assign_x_2 = tf.assign(x, [2])`这句整体移动到了`tf.control_dependencies([tf.assign(x, [2])])`的内部。  
给出的解释是：

> 如果要让`tf.cond()`在其中一个分支中执行命令（如分配），你必须在你要传递给的函数创建执行副命令的操作。  
> If you want to perform a side effect (like an assignment) in one of the branches, you must create the op that performs the side effect inside the function that you pass to .  
> 因为在TensorFlow图中的执行是依次向前流过图形的，所以在任一分支中引用的所有操作必须在条件进行求值之前执行。这意味着true和false分支都接受对`tf.assign()`  op 的控制依赖。  
> Because execution in a TensorFlow graph flows forward through the graph, all operations that you refer to in either branch must execute before the conditional is evaluated. This means that both the true and the false branches receive a control dependency on the  `tf.assign()`  op.

翻译的可能不够准确，大意就是`assign_x_2 = tf.assign(x, [2])`这句话在`tf.cond`已经执行过了，因此无论执行`update_x_2`（让x=2）或`lambda: tf.identity(x)`（保持x不变），得到的结果都是`x=2`。  
这么来看其实是一个很简单的问题，定义时不仅定义了模型，也隐含着定义了执行顺序。



# tf.control_dependencies()
这个函数加不加看起来没有什么区别，比如：  
```python
import tensorflow as tf   
pred = tf.placeholder(tf.bool, shape=[])  
x = tf.Variable([1])  
# x_2 = tf.assign(x, [2])  
def update_x_2():  
 # with tf.control_dependencies([x_2]): #[tf.assign(x, [2])]):  
 return tf.assign(x, [2])  
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))  
with tf.Session() as session:  
 session.run(tf.global_variables_initializer())  
 print(y.eval(feed_dict={pred: False}))  # ==> [1]  
 print(y.eval(feed_dict={pred: True}))   # ==> [2]  
```

去掉之后运行结果和正确的相同。具体作用还是看一下官网吧……  
直接搜`tf.control_dependencies`得到的信息并不多：

> Wrapper for Graph.control_dependencies() using the default graph.  
> See  [`tf.Graph.control_dependencies`](https://www.tensorflow.org/api_docs/python/tf/Graph#control_dependencies)  for more details.

在`tf.Graph.control_dependencies`这里确实讲得很详细，其作用简单来说就是**控制计算顺序**。
```python
with g.control_dependencies([a, b, c]):  
 # `d` and `e` will only run after `a`, `b`, and `c` have executed.  
 d = ...  
 e = ...  
```

有了这句话，`with`中的语句就会在`control_dependencies()`中的操作执行之后运行，并且也支持嵌套操作。在给出的错误例子中，很像开头提出的问题：
```python
# WRONG  
def my_func(pred, tensor):  
 t = tf.matmul(tensor, tensor)  
 with tf.control_dependencies([pred]):  
 # The matmul op is created outside the context, so no control  
 # dependency will be added.  
 return t  
  
# RIGHT  
def my_func(pred, tensor):  
 with tf.control_dependencies([pred]):  
 # The matmul op is created in the context, so a control dependency  
 # will be added.  
 return tf.matmul(tensor, tensor)  
```

上面`t`操作在`tf.control_dependencies`之前已经被执行了，因此就无法控制`t`的先后顺序。如果我们把`my_func`看作是`tf.cond`中的分支操作函数，那么很可能在`pred`更新之前就已经进行了操作，因此可能造成一些错误。

# 总结
这么一看，好像我自己写的没有注意这么多细节，但目前从结果上看好像还都没什么问题，或许需要重新改写一下。


