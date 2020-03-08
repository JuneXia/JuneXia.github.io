---
title: 
date: 2019-9-15
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

# 环境搭建
要求tensorboard版本至少是1.14
```bash
sudo pip3 install tensorboard
```

# tensorboard使用示例
```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()
```

```bash
$ cd to project path
$ tensorboard --logdir=./runs
```


# SummaryWriter使用详解

```python
class SummaryWriter(object):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            log_dir (string): Save directory location. Default is
              runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
              Use hierarchical folder structure to compare
              between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
              for each new experiment to compare across them.
            comment (string): Comment log_dir suffix appended to the default
              ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
            purge_step (int):
              When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
              any events whose global_step larger or equal to :math:`T` will be
              purged and hidden from TensorBoard.
              Note that crashed and resumed experiments should have the same ``log_dir``.
            max_queue (int): Size of the queue for pending events and
              summaries before one of the 'add' calls forces a flush to disk.
              Default is ten items.
            flush_secs (int): How often, in seconds, to flush the
              pending events and summaries to disk. Default is every two minutes.
            filename_suffix (string): Suffix added to all event filenames in
              the log_dir directory. More details on filename construction in
              tensorboard.summary.writer.event_file_writer.EventFileWriter.

        Examples::

            from torch.utils.tensorboard import SummaryWriter

            # create a summary writer with automatically generated folder name.
            writer = SummaryWriter()
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

            # create a summary writer using the specified folder name.
            writer = SummaryWriter("my_experiment")
            # folder location: my_experiment

            # create a summary writer with comment appended.
            writer = SummaryWriter(comment="LR_0.1_BATCH_16")
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/

        """
        torch._C._log_api_usage_once("tensorboard.create.summarywriter")
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix

        ...

    ...

```

**功能**：提供创建event-file的高级接口

- **log_dir**: event-file输出文件夹，如果不指定，则会在当前目录下创建runs文件夹，event-file将被保存到runs目录下的对应文件夹下；如果指定了log_dir，则comment不起作用；
- **comment***: 不指定log_dir时，文件夹后缀
- **filename_suffix**: event-file 文件名后缀

**注**：通常不会使用默认的log_dir，因为开发中一般都要将代码和数据分离。

```python
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# flag = 0
flag = 1
if flag:
    log_dir = "./train_log/test_log_dir"
    # writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix="12345678")
    writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")

    for x in range(100):
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.close()
```

**指定log_dir时生成的文件目录**
```bash
.
├── train_log
│   └── test_log_dir
│       └── events.out.tfevents.1583414749.ailab-server.13132.012345678
```

**不指定log_dir时生成的文件目录**
```bash
.
├── runs
│   └── Mar05_08-26-58_ailab-server_scalars
│       └── events.out.tfevents.1583414821.ailab-server.13260.012345678
```

## 成员方法：add_scalar
```python
class SummaryWriter(object):
    def __init__(...):
        ...
    
    ... 

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %

        """
        if self._check_caffe2_blob(scalar_value):
            scalar_value = workspace.FetchBlob(scalar_value)
        self._get_file_writer().add_summary(
            scalar(tag, scalar_value), global_step, walltime)
```

**功能**：记录标量

- **tag**: 图像标签名，图的唯一标识
- **scalar_value***: 要记录的标量
- **global_step**: x轴


## 成员方法：add_scalars
```python
class SummaryWriter(object):
    def __init__(...):
        ...
    
    ... 

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Adds many scalar data to summary.

        Note that this function also keeps logged scalars in memory. In extreme case it explodes your RAM.

        Args:
            main_tag (string): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag.replace("/", "_") + "_" + tag
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(fw_tag, self.max_queue, self.flush_secs,
                                self.filename_suffix)
                self.all_writers[fw_tag] = fw
            if self._check_caffe2_blob(scalar_value):
                scalar_value = workspace.FetchBlob(scalar_value)
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)
```

**功能**：在同一个图中记录多个标量

- **main_tag**: 该图的标签，等同于add_scalar中的tag
- **tag_scalar_dict***: key是变量的tag，value是变量的值
- **global_step**: x轴

**代码示例：**
```python
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# flag = 0
flag = 1
if flag:

    max_epoch = 100

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(max_epoch):

        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                                 "xcosx": x * np.cos(x)}, x)

    writer.close()
```


## 成员方法：add_histogram
```python
class SummaryWriter(object):
    def __init__(...):
        ...
    
    ... 

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (torch.Tensor, numpy.array, or string/blobname): Values to build histogram
            global_step (int): Global step value to record
            bins (string): One of {'tensorflow','auto', 'fd', ...}. This determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            for i in range(10):
                x = np.random.random(1000)
                writer.add_histogram('distribution centers', x + i, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram.png
           :scale: 50 %

        """
        if self._check_caffe2_blob(values):
            values = workspace.FetchBlob(values)
        if isinstance(bins, six.string_types) and bins == 'tensorflow':
            bins = self.default_bins
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime)
```
**功能**：统计直方图与多分位数折线图，对模型分析模型参数分布以及梯度分布是很有用的。
关于多分位折线图，我的理解是：监控values的数值区间中多个位置的value的变化情况的折线图。

- **tag**: 图像的标签名，图的唯一标识符
- **values**: 要统计的参数
- **global_step**: y轴
- **bins**: 取直方图的bins，通常取默认的“tensorflow”即可。（关于bins，我的理解就是将values的数值区间划分成一个个的bin(箱子)，然后统计落在每个bin的value的个数）


```python
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# flag = 0
flag = 1
if flag:

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(2):

        np.random.seed(x)

        data_union = np.arange(100)
        data_normal = np.random.normal(size=1000)

        writer.add_histogram('distribution union', data_union, x)
        writer.add_histogram('distribution normal', data_normal, x)

        plt.subplot(121).hist(data_union, label="union")
        plt.subplot(122).hist(data_normal, label="normal")
        plt.legend()
        plt.show()

    writer.close()
```

tensorboard(左)和plt(右)画的其中一个epoch的分布图如下：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/tensorboard_histograms.jpg" width = 50% height = 50% /><img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/tensorboard_histograms2.jpg" width = 50% height = 60% />
</div>

可以看到plt画出的normal直方图和tensorboard画出的normal直方图基本类似，但是对于union直方图，正确的应该是plt所画出的图，tensorboard的是由于其内部做了一些显示上的优化。

再来看看tensorboard画出的distribution图（多分位数折线图）：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/tensorboard_distributions.jpg" width = 60% height = 60% />
</div>


# 参考文献
[1] DeepShare.net > PyTorch框架


