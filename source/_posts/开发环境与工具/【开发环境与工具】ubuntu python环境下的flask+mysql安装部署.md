---
title: 【开发环境与工具】ubuntu python环境下的flask+mysql安装部署
date: 2019-04-12
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

## flask 安装
sudo pip3 install flask

## mysql 安装
sudo apt-get install mysql-server
sudo apt-get install mysql-client
sudo apt install libmysqlclient-dev

安装成功后可以通过下面的命令测试是否安装成功：
sudo netstat -tap | grep mysql

检查mysql服务状态：
systemctl status mysql.service

安装mysql  python调用接口
sudo pip3 install pymysql

进入mysql的方式（quit退出）
mysql -uroot -p

进入mysql后的可以通过mysql命令行操作
mysql> show user


## 创建数据库和表
下面开始创建数据库有表，例如可以新建一个名为face_recognization.sql的文件，然后输入一下内容：
```sql
-- 创建数据库
create database if not exists db_face_recognization;
use db_face_recognization;

-- 特征表
drop table if exists tb_face_feature;
create table if not exists tb_face_feature (
    person_id smallint unsigned not null comment "person_id",
    img varchar(256) default null comment "图片路径",
    feature blob default null comment "特征值",
    create_time timestamp not null default current_timestamp comment "创建时间",
    primary key (person_id)
) engine=InnoDB default charset=utf8mb4 comment="人脸特征表";
```

通过下面的命令进入mysql环境：
mysql -uroot -p

在mysql环境下执行下面的命令即可创建数据库和表：
source  /**/pathto/**/face_recognization.sql
注意：该sql文件一定要给全路径


## mysql常用命令

**进入数据库基本操作**
```sql
-- 显示当前用户下的所有数据库
show databases;

-- 使用某个数据库
use db_face_recognization;

-- 显示当前数据库下所有表
mysql> show tables;
+---------------------------------+
| Tables_in_db_face_recognization |
+---------------------------------+
| tb_admin_user                   |
| tb_face_feature                 |
+---------------------------------+
2 rows in set (0.00 sec)

-- 查看表的字段信息
desc tb_face_feature

```


**显示表中的所有字段**
```sql
show columns from tb_face_feature;
```


**添加新字段**[3]
```sql
alter table tb_face_feature add column person_name varchar(128) default null comment "person_name" after person_id;
-- 往tb_face_feature表中增加字段 person_name，长度128，默认为null，备注：person_name，添加在person_id后面
```


**删除字段**
```sql
alter table tb_face_feature drop person_name;
```


**创建表时设置主键自增，且指定自增初始值**
```sql
drop table if exists tb_test2;
create table if not exists tb_test2 (
    person_id smallint unsigned not null auto_increment comment "person_id",
    person_name varchar(128) default null comment "person_name",
    img varchar(256) default null comment "图片路径",
    feature blob default null comment "特征值",
    create_time timestamp not null default current_timestamp comment "创建时间",
    primary key (person_id)
) engine=InnoDB auto_increment=1 default charset=utf8mb4 comment="人脸特征表";

-- 注意上述语句中的auto_increment
```


**已有主键，仅设置主键自增**
```sql
alter table tb_face_feature change person_id person_id smallint unsigned AUTO_INCREMENT;  -- 设置主键自增
-- 改变tb_face_feature表的person_id字段名为person_id, 数据类型为smallint unsigned，自增
alter table tb_face_feature AUTO_INCREMENT=1;  -- 设置主键从1开始自增
```

**修改主键字段名称**
```sql
alter table tb_face_feature change person_id feature_id smallint unsigned auto_increment;
-- 将tb_face_feature表的person_id字段名改为feature_id，并指定其数据类型为smallint unsigned,且数值自增
```

**从一个表导入数据到另一个表**
```sql
insert into tb_person(person_id, person_name, create_time) select feature_id, person_name, create_time from tb_face_feature;
-- 上述语句要求目标表tb_person和源表tb_face_feature相应的数据类型要相同。

```

**将同一个表中的某个字段复制到另一个字段**
```sql
update tb_face_feature set person_id=feature_id
```


**批量插入**
```sql
insert into tb_face_feature(person_name, feature_mark, feature) values('lucy','A',10010),('andy','B',10012);
```

**同时修改多个表**
```sql
update tb_person tbp, tb_face_feature tbf set tbp.person_name='吴三桂', tbf.img='./xxoo/xx00.jpg', tbf.feature=10011 where tbp.person_id=tbf.person_id and tbp.person_id=56 and tbf.feature_mark='A';
```




## 参考文献
[1] [在Ubuntu16.04下安装mysql](https://blog.csdn.net/xiangwanpeng/article/details/54562362)
[2] [Ubuntu18.04 安装MySQL](https://blog.csdn.net/weixx3/article/details/80782479)
[3] [MySQL ALTER命令](https://www.runoob.com/mysql/mysql-alter.html)

