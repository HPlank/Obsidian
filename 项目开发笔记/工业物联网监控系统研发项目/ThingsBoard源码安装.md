
## 1.开发环境

-   官方编译注意事项
    
    **1、JDK 版本采用 11，记得同步 IDEA JDK 设置；**
    
    **2、注释 pom 文件 license 检查；**
    
    **3、Node 版本在 12~14 之间；**
    
    **4、安装插件 IntelliJ Lombok plugin 和 Protobuf Support；**
    
    **5、Maven 跟 NPM 都使用官方镜像；**
    
    **6、能有良好的网速，能够快速的访问 dockerhub、github 等等国外网站。**
    
    **以上准备工作做完了，使用以下命令进行编译，一次没过大部分都是网络问题，多试几次**
    
-   JDK111（安装时jdk等环境随项目版本升级改变请查看社区[Thingsboard · 社区 · 物联网技术社区-物联网平台-ThingsBoard (iotschool.com)](https://www.iotschool.com/topics/node8)）
    
-   maven3.6+
    
-   Git
    
-   编译下载pom包需要VPN外网环境
    
-   nodejs 16.3+
    
-   IDEA（需要提前将idea内存调大，防止项目打开较慢）
    
-   手动下载 yarn
    
    `npm install -g yarn yarn config set registry https://registry.npm.taobao.org -g yarn config set sass_binary_site http://cdn.npm.taobao.org/dist/node-sass -g`
    
-   maven镜像
```
    <mirror>  
          <id>nexus-public-snapshots</id>  
          <mirrorOf>public-snapshots</mirrorOf>  
         <url>http://maven.aliyun.com/nexus/content/repositories/snapshots/</url>  
    </mirror>  
    <mirror>  
          <id>nexus</id>  
          <name>internal nexus repository</name>  
          <url>https://repo.maven.apache.org/maven2</url>  
          <mirrorOf>central</mirrorOf>  
    </mirror>  
    <mirror>  
      <id>maven-central</id>  
      <name>central</name>  
      <url>https://repo1.maven.org/maven2/</url>  
      <mirrorOf>central</mirrorOf>  
    </mirror>  
    ​  
    <mirror>  
      <id>uk</id>  
      <mirrorOf>central</mirrorOf>  
      <name>Human Readable Name for this Mirror.</name>  
      <url>http://uk.maven.org/maven2/</url>  
    </mirror>  
    ​  
    <mirror>  
      <id>CN</id>  
      <name>OSChina Central</name>  
      <url>http://maven.oschina.net/content/groups/public/</url>  
      <mirrorOf>central</mirrorOf>  
    </mirror>  
    ​  
     <mirror>           
      <id>central</id>           
      <name>aliyun central</name>           
      <url>https://maven.aliyun.com/repository/central</url>          
      <mirrorOf>central</mirrorOf>  
    </mirror>  
    <mirror>           
      <id>google</id>           
      <name>aliyun google</name>           
      <url>https://maven.aliyun.com/repository/google</url>          
      <mirrorOf>google</mirrorOf>  
    </mirror>  
    <mirror>           
      <id>public</id>           
      <name>aliyun public</name>           
      <url>https://maven.aliyun.com/repository/public</url>          
      <mirrorOf>public</mirrorOf>  
    </mirror>  
    <mirror>           
      <id>gradle-plugin</id>           
      <name>aliyun gradle-plugin</name>           
      <url>https://maven.aliyun.com/repository/gradle-plugin</url>          
      <mirrorOf>gradle-plugin</mirrorOf>  
    </mirror>  
    <mirror>           
      <id>spring</id>           
      <name>aliyun spring</name>           
      <url>https://maven.aliyun.com/repository/spring</url>          
      <mirrorOf>spring</mirrorOf>  
    </mirror>  
    <mirror>           
      <id>spring-plugin</id>           
      <name>aliyun spring-plugin</name>           
      <url>https://maven.aliyun.com/repository/spring-plugin</url>          
      <mirrorOf>spring-plugin</mirrorOf>  
    </mirror>   
    <mirror>           
      <id>grails-core</id>           
      <name>aliyun grails-core</name>           
      <url>https://maven.aliyun.com/repository/grails-core</url>          
      <mirrorOf>grails-core</mirrorOf>  
    </mirror>  
    <mirror>           
      <id>apache-snapshots</id>           
      <name>aliyun apache-snapshots</name>           
      <url>https://maven.aliyun.com/repository/apache-snapshots</url>          
      <mirrorOf>apache-snapshots</mirrorOf>  
    </mirror>
```

    

## 2.源码拉取

ThingsBoard官网 ---> 关于我们 ---> 博客 ---> 拉到最下方找到GIT点击进入Git源码

![image-20220729112016628](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220729112016628.png?lastModify=1666354911)

![image-20220729112037974](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220729112037974.png?lastModify=1666354911)

使用git拉取如果速度太慢可以使用Gitee通过github链接导入本地仓库，使用gitee下载链接下载

![image-20220729113552476](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220729113552476.png?lastModify=1666354911)

## 3.编译安装（需一小时左右，网络问题等可能中断）

### 1.编译命令：

`mvn clean install -DskipTests --settings maven配置文件目录`

`mvn clean install -DskipTests --settings D:\Dsoft\apache-maven-3.8.6\conf\thingsboardSettings.xml`

### 2.在项目路径下执行编译指令

![](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220729153538839.png?lastModify=1666354911)

### 3.中断解决方法

![image-20220730103326855](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730103326855.png?lastModify=1666354911)

删除 ui-ngx\patches目录下的geoman-io+leaflet-geoman-free+2.11.4.patch文件后就可以正常编译

如遇nodejs下载失败，手动下载所需的node版本到指定目录，注意node命名与提示要相同

如果提示maven与项目内不匹配，需要改成指定版本maven

![image-20220730112342627](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730112342627.png?lastModify=1666354911)

安装成功

## 4数据库下载

postgreSQL数据库下载，可自定义安装目录，除下图列出，默认next，端口等选项默认安装![image-20220730113617012](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730113617012.png?lastModify=1666354911)

安装图中所选项继续。

![image-20220730113809761](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730113809761.png?lastModify=1666354911)

![image-20220730113822115](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730113822115.png?lastModify=1666354911)

需要下载语言包，保证网络连接。

![image-20220730113854819](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730113854819.png?lastModify=1666354911)

弹出安装应用程序界面，选择安装语言包，点下一个：

![image-20220730113920980](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730113920980.png?lastModify=1666354911)

![image-20220730113943027](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730113943027.png?lastModify=1666354911)

![image-20220730114023230](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730114023230.png?lastModify=1666354911)

默认next安装即可。

![image-20220730114008603](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730114008603.png?lastModify=1666354911)

可使用navicat等数据库连接工具连接。

创建thingsboard数据库

![image-20220729150940011](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220729150940011.png?lastModify=1666354911)

## 5.导入Idea

-   提前查看idea的maven仓库是否是之前编译所使用的仓库位置
    
    ![image-20220730120338376](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730120338376.png?lastModify=1666354911)
    
-   导入项目，检查idea中使用的jdk版本是否正确
    

![image-20220730120726634](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730120726634.png?lastModify=1666354911)

![image-20220730120754960](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730120754960.png?lastModify=1666354911)

![image-20220730125906484](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730125906484.png?lastModify=1666354911)

![image-20220730125919111](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730125919111.png?lastModify=1666354911)

-   修改项目中数据库连接信息，用户名和密码
    

![image-20220729151518933](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220729151518933.png?lastModify=1666354911)

-   向数据库中导入数据
    
-   打开application-target-Windows所在目录
    

![image-20220730130356413](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730130356413.png?lastModify=1666354911)

![image-20220730130449184](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730130449184.png?lastModify=1666354911)

-   执行脚本 install_dev_db.bat
    

![image-20220730130608197](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730130608197.png?lastModify=1666354911)

![image-20220730130703686](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730130703686.png?lastModify=1666354911)

-   查看数据库中数据
    
    ![image-20220730130746228](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730130746228.png?lastModify=1666354911)
    
-   启动项目，访问localhost:8080
    

![image-20220730131624345](file://C:/Users/78749/AppData/Roaming/Typora/typora-user-images/image-20220730131624345.png?lastModify=1666354911)

-   **系统管理员**: [sysadmin@thingsboard.org](mailto:sysadmin@thingsboard.org) / sysadmin
    
-   **租户管理员**: [tenant@thingsboard.org](mailto:tenant@thingsboard.org) / tenant
    
-   **客户**: [customer@thingsboard.org](mailto:customer@thingsboard.org) / customer