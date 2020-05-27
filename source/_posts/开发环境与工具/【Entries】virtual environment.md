---
title: 
date: 2020-05-26
tags:
categories: ["开发环境与工具"]
mathjax: true
---

&emsp; 在python开发中，我们可能会遇到一种情况，就是当前的项目依赖的是某一个版本，但是另一个项目依赖的是另一个版本，这样就会造成依赖冲突，而virtualenv就是解决这种情况的，virtualenv通过创建一个虚拟化的python运行环境，将我们所需的依赖安装进去的，不同项目之间相互不干扰。
<!--more-->


# 安装 virtualenv
```bash
sudo pip install virtualenv

# 或者

sudo apt-get install virtualenv
```


# 创建虚拟环境

## 创建虚拟环境-方法1
```bash
xiajun@xj-ubuntu:~$ mkvirtualenv tmp-venv

# output log
Using base prefix '/usr'
New python executable in /home/xiajun/.virtualenvs/tmp-venv/bin/python3
Also creating executable in /home/xiajun/.virtualenvs/tmp-venv/bin/python
Installing setuptools, pip, wheel...done.
virtualenvwrapper.user_scripts creating /home/xiajun/.virtualenvs/tmp-venv/bin/predeactivate
virtualenvwrapper.user_scripts creating /home/xiajun/.virtualenvs/tmp-venv/bin/postdeactivate
virtualenvwrapper.user_scripts creating /home/xiajun/.virtualenvs/tmp-venv/bin/preactivate
virtualenvwrapper.user_scripts creating /home/xiajun/.virtualenvs/tmp-venv/bin/postactivate
virtualenvwrapper.user_scripts creating /home/xiajun/.virtualenvs/tmp-venv/bin/get_env_details

(tmp-venv) xiajun@xj-ubuntu:~$ 
```

> 注意：\
> 1. mkvirtualenv 创建的虚拟环境目录位于 /home/username/.virtualenvs 目录下；
> 2. mkvirtualenv 创建好虚拟环境后，会直接进入该环境。


## 创建虚拟环境-方法2
```bash
xiajun@xj-ubuntu:~$ virtualenv tmp2-venv

# output log
Using base prefix '/usr'
New python executable in /home/xiajun/tmp2-venv/bin/python3
Also creating executable in /home/xiajun/tmp2-venv/bin/python
Installing setuptools, pip, wheel...done.

# 激活虚拟环境
xiajun@xj-ubuntu:~$ source tmp2-venv/bin/activate

(tmp2-venv) xiajun@xj-ubuntu:~$
```

> 注意：\
> 1. virtualenv 创建的虚拟环境目录位于 virtualenv 命令的执行目录下；
> 2. virtualenv 创建好虚拟环境后，不会直接进入该环境，需要执行 `source tmp2-venv/bin/activate` 命令进入该虚拟环境。


## 创建虚拟环境-方法3
-p or --python 这个选项用来指定解释器的位置 ， 如果 本地装了多个python 版本，要用这个指定一下 用哪个解释器来生成虚拟环境。

```bash
xiajun@xj-ubuntu:~$ which python3

# output log
/usr/bin/python3
xiajun@xj-ubuntu:~$ virtualenv -p /usr/bin/python3 py3venv
Already using interpreter /usr/bin/python3
Using base prefix '/usr'
New python executable in /home/xiajun/py3venv/bin/python3
Also creating executable in /home/xiajun/py3venv/bin/python
Installing setuptools, pip, wheel...done.
xiajun@xj-ubuntu:~$ source py3venv/bin/activate

(py3venv) xiajun@xj-ubuntu:~$ 
```


# 退出虚拟环境
```bash
deactivate
```


# 总结
&emsp; python 的虚拟环境其实有很多。 官方推荐就是使用 lib/venv 这个模块来生成的 python 虚拟环境。当然第三方也有很多方案。比如 conda, pipenv, pyenv, poetry 等等。
pyenv 就不要用了。引用官方的一句话：3.6 版后已移除: pyvenv 是 Python 3.3 和 3.4 中创建虚拟环境的推荐工具，不过在 Python 3.6 中已弃用。


# 参考文献
[使用 virtualenv 创建虚拟环境](https://blog.csdn.net/u010339879/article/details/101519007)
