**载荷测量**
基于面积比，通过比较载荷在输送带上所占的面积与整个视场的比值来实现
主要采用语义分割技术来实现，即从图像中分割或确定属于负载类的像素，对像素数进行计数，然后计算计数值与整个屏幕的像素数的比值，或者与输送机满载时属于负载的像素数的比值。
忽略了高度信息，只能对运输量进行模糊估计

**输送带的偏移状态的监测**
用目标检测框的对角线信息来表示传送带的边缘信息，在特定标签的引导下完成传送带边缘区域的检测

基本架构：
主干表示特征提取网络，其通过卷积不断地从输入图像中提取特征
颈部表示特征增强网络，其通过特征融合来丰富提取的特征信息，有助于提高检测或分类的准确性，而头部是网络的最终预测部分。
生成的预测信息可以用于解码和计算预测误差。
SPPF模块本质上是空间金字塔池（SPP）网络的一种变体，可以实现不同尺度特征信息的融合，对提高网络的预测精度有一定的积极作用

## CF-YOLO: Cross Fusion YOLO for Object Detection in Adverse Weather With a High-Quality Real Snow Dataset
i）我们收集了一个高质量的户外数据集，称为RSOD，用于现实世界下雪场景中的对象检测。RSOD包含2100张以COCO和YOLO格式标注的真实世界雪景图像（带有行人、汽车、交通灯等标记）
ii）我们通过引入一个称为雪覆盖率（SCR）的指标奋进定量评估雪对每个物体的影响。为了计算SCR，我们开发了一种无监督训练策略来训练一个简单而有效的CNN模型，具有独特的激活函数，称为峰值动作。
我们提出了一种即插即用的交叉融合（CF）块。CF块同时聚合了来自主干不同阶段的特征。这种直接融合方式允许恢复在高级特征提取期间丢失的低级信息。通过用CF块替换YOLOv 5s的颈部。
我们提供了一个真实世界的雪OD数据集（RSOD），它以COCO和YOLO格式进行标记。
引入了一个名为雪覆盖率（SCR）的指标，并开发了一种无监督训练策略来训练CNN模型，该模型具有独特的激活函数，称为Peak Act
用户可以自定义不同网络中的阶段数量、CF块数量和内核大小，以优化模型的性能。·我们提出了一种轻量级且有效的CF-YOLO，以促进下雪OD应用，使得许多户外视觉系统（例如，自动驾驶、监控）可以在下雪天气下平稳运行。

## 基于 YOLOv5s−SDE 的带式输送机煤矸目标检测
YOLOv5s−SDE 在 Backbone 部分添加了 1 个压缩和激励（Squeeze-andExcitation，SE）模块，以提升煤矸小目标检测效果； 将 4 个普通卷积替换为深度可分离卷积 DwConv，以 显著减少参数量和计算量；采用 EIoU（Efficient-IoU） 替换 YOLOv5s 的 CIoU，以提升检测精度。
SE 是一种通道注意力机制，其目的是学习每个 通道的重要程度，增强有用特征，抑制无用特征。
SE 模块主要由压缩、激励及特征图标定 3 个部分组 成，
![[Pasted image 20230711100951.png]]
压缩操作采用全局平均池化将整个通道上大小 为 W×H×C（W，H 分别为宽度和高度，C 为通道数）的 特征图矩阵压缩成 1×1×C 的向量 Z
激励操作将全连接层作用于特征图，对每个通 道的重要性进行预测，再将得到的重要性权重作用 到相应通道上，构建通道之间的相关性。
S σ(·) w1 w2 δ(·) 式中 ： 为激励操作后所得向量 ； 为 Sigmoid 函 数； 和 分别为 2 个全连接层的降维和升维权重； 为非线性激活函数。

在预测框与真实框不相交的情况下， IoU 无法准确描述二者的距离信息。 CIoU 不但考 虑了边界框的重叠面积和中心点距离，而且增加了 高宽比惩罚项，目前 YOLOv5s 采用 CIoU 函数作为 边界框回归损失函数。EIoU 在 CIoU 基础上改进了 宽高损失部分，使目标框与锚盒的宽度和高度之差 最小，收敛速度更快。
在预测框与真实框不相交的情况下， IoU 无法准确描述二者的距离信息。 CIoU 不但考 虑了边界框的重叠面积和中心点距离，而且增加了 高宽比惩罚项，目前 YOLOv5s 采用 CIoU 函数作为 边界框回归损失函数。EIoU 在 CIoU 基础上改进了 宽高损失部分，使目标框与锚盒的宽度和高度之差 最小，收敛速度更快。
### 基于改进 Mask R−CNN 的刮板输送机 铁质异物多目标检测
