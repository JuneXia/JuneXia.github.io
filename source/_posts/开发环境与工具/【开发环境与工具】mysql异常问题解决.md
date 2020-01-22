---
title: 【开发环境与工具】mysql异常问题解决
date: 2019-05-14
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

# mysql 异常问题解决
## 异常1
```bash
~$ mysql -u root -p
Enter password:
ERROR 1698 (28000): Access denied for user 'root'@'localhost'
```

解决方案：
```bash
# 试了诸多方法，最后发现直接以下面这条命令就能进入mysql
~$ mysql -u root


# 进入mysql后，设置root用户的密码

# 查看一下user表，错误的起因就是在这里， root的plugin被修改成了auth_socket，用密码登陆的plugin应该是mysql_native_password。
mysql> select user, plugin from mysql.user; 
+-----------+-----------------------+
| user      | plugin                |
+-----------+-----------------------+
| root      | auth_socket           |
| mysql.sys | mysql_native_password |
| dev       | mysql_native_password |
+-----------+-----------------------+
3 rows in set (0.01 sec)

# 解决办法1: 参考文献[1]
mysql> USE mysql;
mysql> UPDATE user SET plugin='mysql_native_password' WHERE User='root';
mysql> FLUSH PRIVILEGES;
mysql> exit;

$ service mysql restart
```

## 异常2
```bash
~$ mysql -u root -p
Enter password: 
ERROR 1045 (28000): Access denied for user 'root'@'localhost' (using password: YES)
```

```bash
~$ mysql -u root
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 2 Server version: 5.7.10 MySQL Community Server (GPL)

# 设置root用户密码，参考文献[2]
mysql> update mysql.user set authentication_string=PASSWORD('newPwd'), plugin='mysql_native_password' where user='root';
Query OK, 1 row affected, 1 warning (0.00 sec)
Rows matched: 1  Changed: 1  Warnings: 1 mysql> flush privileges;
Query OK, 0 rows affected (0.00 sec)
```

# 参考文献
[1] [解决 MySQL 的 ERROR 1698 (28000): Access denied for user 'root'@'localhost'](https://blog.csdn.net/jlu16/article/details/82809937)
[2] [Mysql ERROR 1698 (28000) 解决](https://blog.csdn.net/qq_34771403/article/details/73927962)
