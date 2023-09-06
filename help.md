创建项目 django-admin startproject mysite
创建app  python manage.py startapp app01 
命令行启动 python manage.py runserver 0.0.0.0:5000

数据库
启动 mysql -u root -p
创建 create database gx01 DEFAULT CHARSET utf8 COLLATE utf8_general_ci;
展示 show databases;
使用 use gx01;
展示 show tables;
查询 select * from app01_admin;

迁移
python manage.py makemigrations
python manage.py migrate  