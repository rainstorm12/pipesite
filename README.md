# pipesite backend

## 运行
创建项目 django-admin startproject mysite

创建app  python manage.py startapp app01 

命令行启动 python manage.py runserver 0.0.0.0:5000

## 数据库
### mysql使用
启动 mysql -u root -p

创建 create database gx01 DEFAULT CHARSET utf8 COLLATE utf8_general_ci;

展示 show databases;

使用 use gx01;

展示 show tables;

查询 select * from app01_admin;

新增 INSERT INTO table_name (column1, column2, column3, ...) VALUES (value1, value2, value3, ...);

### mysql迁移
python manage.py makemigrations

python manage.py migrate  
