---
title: 
date: 2017-11-19
tags:
categories: ["TensorFlow笔记"]
mathjax: true
---

## 方法1：导入整张Graph
当模型graph和预训练模型graph是一样的时候，我们通常可以import整张graph来恢复模型(即复用预训练模型)。

这种情况是：通常是预训练模型是我们自己训练的，再次微调时可以使用这种方法。文献[1]中对该方法也有所介绍。
<!-- more -->

代码示例1：
```python
checkpoint_path = 'path_to_pretrain_model'
network = create_alexnet(...)

saver = tf.train.Saver()  # 用于保存新的模型
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    restore_saver = tf.train.import_meta_graph(checkpoint_path + 'model_name.ckpt.meta')
    restore_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
```



## 恢复指定层
参考文献[2]第1.2节。

代码示例2：
```python
checkpoint_path = 'path_to_pretrain_model'
network = create_alexnet(...)

# OK *************************************
# var_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv[12345]|fc[67]')
# restore_saver = tf.train.Saver(var_to_restore)
# *************************************

# OK *************************************
var = tf.global_variables()
var_to_restore = [val for val in var if val.name.split('/')[0] not in skip_layer]  # 除了skip_layer中的层，剩下的都restore
restore_saver = tf.train.Saver(var_to_restore)
# *************************************

saver = tf.train.Saver()  # 用于保存新的模型
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    restore_saver.restore(sess, os.path.join(checkpoint_path, 'model_name.ckpt'))
    # tf.initialize_variables(var_to_restore)  # 有没有都ok

    for step in range(training_epoch):
        sess.run(train_op, feed_dict={...})

```


## 训练指定层
参考文献[2]第3节“冻结较低层”。




## 参考文献
[1] 【深度学习笔记1.1】人工神经网络
[2] 【深度学习笔记1.3】复用预训练层
