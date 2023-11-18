#### 创建项目
```
# 在终端中进入项目文件夹创建
"python文件夹下scripts\django-admin.exe" startproject 项目名称
# 如果以及加入环境变量
django-admin startproject 项目名称
```

##### 注册app 【setting.py】
在setting—》 installed_apps —》app名称 . apps . 方法名

##### 编写URL和视图函数对应关系【urls.py】
```
from django.urls import path, include  

from news import views  
  
  
urlpatterns = [  
    # path('admin/', admin.site.urls),  
  
    # www.xxx.com/index/ => 函数  
    path('index/', views.index()),  
]
```
##### 编写views视图函数
```  
# Create your views here.  
  
from django.shortcuts import render, HttpResponse  
  
  
def index(request):  
    return HttpResponse("欢迎使用")

```
#### 启动
```
python manager.py runserver
```

#### django连接数据库
创建django数据库
```
pip install mysqlclient
```


#### 在setting.py文件中进行配置和修改
``` 
DATABASES = {  
    'default':{  
        'EENGINE': 'django.db.backends.mysql',  
        'NAME':'mysql', # 数据库名称  
        'USER': 'root',  
        'PASSWORD': 'mysql',  
        'HOST': '127.0.0.1', # 那台机器安装了Mysql  
        'PORT': '3306',  
    }  
}
```

#### django操作表
- 创建表    在models.py文件中
```
from django.db import models  
  
# Create your models here.  
  
class UserInfo(models.Model):  
    name = models.CharField(max_length=32)  
    password =  models.CharField(max_length=64)  
    age = models.IntegerField()  
  
"""  
create table news_UserInfo(  
    id bigint auto_increment primary key,    name varchar(32),    password varchar(64),    age int)  
"""

```
执行：
```
 python manage.py makemigrations

 python manage.py migrate


```

- 删除表
- 修改表
在新增列时，由于已存在的列中可能已有数据，所以新增列必须要指定新增列对应的数据:
- 1.  手动输入一个值
- 设置默认值
	age = models.IntegerField(default=2)
- 允许为空
	data = models.IntegerField(null=True, blank=True)

### 数据库
- 【加速查找，允许数据冗余】
	当查询次数请求量大时，取消连表操作

无约束：
depart_id = models.BigIntegerField(version_name="部门ID")

1. 有约束：
-to：与哪张表关联
-to_field，表中的那一列关联
2. django自动
-写的depart，生成数据列depart_id
3. 部门表被删除
	3.1 级联删除
	depart = models.ForeignKey(to = "Department", to_field = “id”, on_delete=models.CASCADE)
	3.2 置空
	depart = models.ForeignKey(to = "Department", to_field = “id”,null = True, blank = True, on_delete=models.SET_NULL)

在django中做约束
gender_choices = (
	（1，"男"），
	（2，"女"），
)

#### Models关键字
<th>
	<tr></tr>
</th>
<th>

</th>
<tr>
<th background-color="#00FF00">字段名</th>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<th>作用</th> <br/>
<th>AutoField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>自增一个IntegerField，根据可用的 ID 自动递增</th> <br/>
<th>BooleanField</th>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>该字段的默认表单部件是checkbox,默认值是 None</th> <br/>
<th>CharField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>一个字符串字段</th> <br/>
<th>DateField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>一个日期，在 Python 中用一个 `datetime.date` 实例表示</th> <br/>
<th>DateTimeField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>一个日期和时间，在 Python 中用一个 `datetime.datetime` 实例表示</th> <br/>
<th>FloatField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>在 Python 中用一个 `float` 实例表示的浮点数</th> <br/>
<th>SmallIntegerField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>就是一个 IntegerField， `-32768` 到 `32767` 的值</th> <br/>
<th>IntegerField</th>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>一个整数。从 `-2147483648` 到 `2147483647` 的值</th> <br/>
<th>TextField</th> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<th>一个大的文本字段。该字段的默认表单部件是一个Textarea</th> <br/>
</tr>


Q 对象 (django.db.models.Q) 对象用于封装一组关键字参数，可以使用 & 和 | 操作符组合起来，当一个操作符在两个Q 对象上使用时，它产生一个新的Q 对象。


```
from food_app.models import Cook
# 查询等级为5的数据
Cook.objects.filter(level=5) 
# 查询等级为5，并且派系为川菜的数据
Cook.objects.filter(level=5,sect="川菜")
# 查询等级为5，并且派系为川菜的数据
Cook.objects.filter(level=5).filter(sect="川菜")
# 查询等级为6，或者派系为湘菜的数据，不支持这个写法！！！
Cook.objects.filter(level=6 or sect="湘菜") 
# 查询等级为6，或者派系为湘菜的数据，不支持这个写法！！！
Cook.objects.filter(level=6 | sect="湘菜")) 


from django.db.models import Q
# 查询等级为6，或者派系为湘菜的数据
Cook.objects.filter(Q(level=6) | Q(sect="湘菜"))
# 查询等级为6，并且派系为湘菜的数据
Cook.objects.filter(level=6,sect="湘菜")
# 查询等级4，并且等级为6，或者派系为湘菜的数据
Cook.objects.filter(Q(id=4),Q(level=6) | Q(sect="湘菜"))
# 查询等级不为4，并且等级为6，或者派系为湘菜的数据
Cook.objects.filter(~Q(id=4),Q(level=6) | Q(sect="湘菜")) 
# 查询等级4，并且等级为6，或者派系为湘菜的数据
Cook.objects.filter(Q(level=6) | Q(sect="湘菜"),id=4) 
# 查询等级4，并且等级为6，或者派系为湘菜的数据，不支持这个写法！！！
Cook.objects.filter(id=4,Q(level=6) | Q(sect="湘菜")) 
```


## F对象

作用：模型的属性名出现在操作符的右边，就使用F对象进行包裹
```
# 获取工龄等于优秀表现次数的员工
Salary.objects.filter(seniority=F("outstand"))

# 获取平均每年大于1次的员工
Salary.objects.filter(outstand__gt=F("seniority")+1)
# 更新员工工作年限+1
Salary.objects.update(seniority=F('seniority')+1)
```



