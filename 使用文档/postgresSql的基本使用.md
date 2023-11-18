## PostSQL 的安装
```
# install wget if not already installed:
sudo apt install -y wget

# import the repository signing key:
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# add repository contents to your system:
RELEASE=$(lsb_release -cs)
echo "deb http://apt.postgresql.org/pub/repos/apt/ ${RELEASE}"-pgdg main | sudo tee  /etc/apt/sources.list.d/pgdg.list

# install and launch the postgresql service:
sudo apt update
sudo apt -y install postgresql-12
sudo service postgresql start
```


## PostSQL 创建账号并分配数据库访问权限

切换到可以操作plsql数据库的用户：
```bash
su - postgres
```

## 1. 连接到数据库

```javascript
psql -U postgres -d postgres
```
psql: 登录数据库的客户端
-U : 指定登录的用户
-d : 指定登录的数据库

## 2. 创建数据库


```javascript
CREATE DATABASE thingsboard;
```
thingsboard：要创建的数据库名字

```javascript
CREATE USER thingsboard WITH PASSWORD 'thingsboard';
```
新创建的用户需要给登录权限，以初始用户登录，赋予登录权限:

```javascript
ALTER ROLE "asunotest" WITH LOGIN;
```
给用户分配数据库

```bash
GRANT ALL PRIVILEGES ON DATABASE  thingsboard TO thingsboard;
```
刷新权限

```bash
ALTER DATABASE thingsboard OWNER TO thingsboard;
```
查询所有数据库
```
\l
```
进入数据库  
```
 \c + 数据库名称
```

退出登录

```
\q
```
q: quite

## PostSQL 导入导出数据库

**1.导出数据库：**

- 方式一：pg_dump  -U  postgres( -h localhost)  -f  c:\db.sql postgis
* 方式二：pg_dump  -U postgres (-h localhost)    postgis > c:\db.sql

**2.导入数据库：**

* 方式一：psql  -d  postgis  (-h localhost)  -f  c:\db.sql  postgres

**3.导出具体表：**

* 方式一：pg_dump -U postgres -t mytable(-h localhost)  -f  dump.sql  postgis
* 方式二：pg_dump  -U postgres -t mytable (-h localhost)    postgis > c:\db.sql

**4.导入具体表：**

* 方式一：psql  -d         postgis  ( -h localhost -f)  c:\ dump.sql postgres

**参数：**
       postgres：用户

       postgis：数据库名称

       mytable：表名称

        -f, --file=文件名： 输出文件名

        -U, --username=名字：以指定的数据库用户联接

