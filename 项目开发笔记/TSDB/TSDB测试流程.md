Windows:

1、配置Java环境，32位，需要将四个.dll全部放入/jdk/jre/bin

2、解压flume，修改配置文件

3、安装mysql进行odbc测试

4、测试TCP：修改配置文件将source改为TCPSource，运行Flume启动命令，建立Telnet连接进行测试

5、测试COM：运行虚拟串口助手，模拟两个串口COM1和COM2，运行串口调试助手连接COM1，修改com.properites配置文件修改COM2配置信息，运行jar包，修改flume配置文件将source改为COMSource，运行Flume启动命令。关闭时先关串口调试助手的数据发送（COM1）,再关jar包的ComAgent，再关flume，再关虚拟串口

6、测试OPCUA：打开OPCUA模拟软件，修改配置文件将source改为OPCUASource，第一次运行flume，生成客户端和密钥，将模拟软件中Certificates目录下与flume配置的OPCUASource.appLocationName名字相同的密钥设为Trusted，再运行flume，采集到数据。关闭时先关闭flume后关闭模拟软件

7、测试ODBC：使用64位dll测试成功，使用32位dll测试失败。打开mysql服务，创建表，表中要有足够的数据，修改flume配置文件为ODBCSource，主要是sql语句，语句中精确到列，只能包含数值，运行flume会按行读取拿到的列数据

8、测试Modbus：ModbusTCP使用模拟软件模拟数据，修改flume配置文件为ModbusSource，指定参数，运行flume即可获取数据；ModbusRTU先使用虚拟串口助手设置串口，打开Modbus模拟软件配置RTU的COM口信息，修改ModbusSource指定参数，运行flume即可获取数据

9、opc-da

Flume启动命令：bin\flume-ng agent --conf conf --conf-file conf/TSDBReadAgent.conf --name TSDBReadAgent -property flume.root.logger=INFO,console

Ubuntu:

1、安装curl、make、build-essential（gcc）

2、将install文件拷贝到待安装主机
![[Pasted image 20221123172243.png]]install-standlone.sh中 第一行路径与内容中路径（修改两个位置）

![[Pasted image 20221123172943.png]]install-distributed.sh中第一行与内容中路径（修改两个位置）
![[Pasted image 20221123173127.png]]install-distributed.sh

3、修改install-standlone.sh（2）、install-distributed.sh（2）、runserver.sh（1）、opentsdb.conf（1）中路径

4、执行脚本 `bash install-standlone.sh`

5、进入TSDB目录，执行`bash runserver.sh`启动TSDB

6、进入redis文件目录下 redis.conf 修改redis中daemonize字段改为no

7、启动redis，命令：./redis-server ../redis.conf
端口占用问题 ps aux | grep redis   kill -9 端口

8、启动前端服务  apache-tomcat下bin    start.sh

9、数据展示界面登陆访问： [http://localhost:8080/dist/index.html](http://localhost:8080/dist/index.html)（功能正常）

10、设备控制和备份界面： [http://localhost:8080/equ_index](http://localhost:8080/equ_index)（功能正常）

11、修改数据生成脚本MakeData.java路径，在TSDB下新建/Data文件夹，生成测试数据 javac MakeData.java    java MakeData


12、写入速度测试执行 opentsdb.conf 中改auto_metric : true。重启重启opentsdb，bash runserve.sh  ：writespeed_test.sh，需要注意线程数2和文件数6（writespeed_test.sh）

13、查询速度测试：~~需要修改~~~~search.sh~~~~中的时间戳~~  执行：search.sh  time_starttransfer为查询用时

14、 压缩比测试执行：compress_test.sh  执行完成后会强制停止hbase，将缓存区的数据刷入磁盘，tsdb服务会同时受到影响
stop_hbase.sh
