---
title: 【开发环境与工具】MySQL Workbench连接服务器数据库
date: 2019-11-24
tags:
categories: ["开发环境与工具"]
mathjax: true
---

# 连接远程数据库
参考文献[1]
远程服务器上的数据默认是不允许被其他机器访问的，这时候需要开放远程服务器远程连接权限，具体操作步骤如下：
<!-- more -->

1. 在远程服务器上，以root用户进入数据库
```bash
mysql -uroot -p
```

2. 开启服务器mysql远程连接权限
```bash
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'yourpassword' WITH GRANT OPTION;

FLUSH PRIVILEGES;
```
如果限制特定ip连接，将’root’@’%’中 %替换为ip。


3. 修改mysql配置文件/etc/mysql/mysql.conf.d/mysqld.cnf
将其中
```bash
bind-address        = 127.0.0.1
```

改为
```bash
bind-address        = 0.0.0.0
```

4. 最后重启服务即可
```bash
sudo /etc/init.d/mysql restart
```


# 参考文献
[1] [使用MySQL Workbench远程连接Ubuntu MySQL](https://blog.csdn.net/qq_37299249/article/details/72802309)

