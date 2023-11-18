![[Pasted image 20230417102458.png]]
one-stage:   单阶段，只需要知道四个预测结果，一个cnn网络直接做一个回归不需要做额外映衬   直接通过卷积[神经网络](https://so.csdn.net/so/search?q=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&spm=1001.2101.3001.7020)提取特征，预测目标的分类与定位；
速度快 适合实时检测任务 效果不太好
two-stage:  faster-rcnn mask-rcnn 先经过预选后选出特征突出的进行训练 加入区域建议网络  主要思路：直接通过卷积[神经网络](https://so.csdn.net/so/search?q=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&spm=1001.2101.3001.7020)提取特征，预测目标的分类与定位；
![[Pasted image 20230417103345.png]]

MAP指标： 综合衡量检测效果，
FPS：训练速度
IOU：真实值于预测值的 交集/并集 越高标识重合率越高

精度 precision =  TP / (TP+FP)   判为正的样本中 正的概率
召回率： recall= TP / (TP + FN)  真正的正类被识别出的概率

TP : 正类判定为正
FP: 负类判定为正类
FN：正类判定为负类
TN： 负类判定为正类
第一个表示判断是否正确 第二个为判断为正类还是负类
    
核心思想L: 将图片分割为s * s 的格子，然后分别以每个格子为中心点预测是否为被预测物体

**目标检测** **就是要用矩形框把图片中感兴趣的物体框选出来**

### 目标检测的框架：
![[Pasted image 20230306110624.png]]
**Backbone network**，即**主干网络**，是目标检测网络最为核心的部分，大多数时候，backbone选择的好坏，对检测性能影响是十分巨大的。 
**Neck network**，即**颈部网络**，Neck部分的主要作用就是将由backbone输出的特征进行整合。其整合方式有很多，最为常见的就是FPN（Feature Pyramid Network），有关FPN的内容，我们会在展开介绍Neck的时候再次提到的。
**Detection head**，即**检测头**，这一部分的作用就没什么特殊的含义了，就是若干卷积层进行预测，也有些工作里把head部分称为decoder（解码器）的，这种称呼不无道理，head部分就是在由前面网络输出的特征上去进行预测，约等于是从这些信息里解耦出来图像中物体的类别和位置信息。

##### 1.Backbone：目标检测网络的主体结构
为了实现从图像中检测目标的位置和类别，我们会先从图像中提取出些必要的特征信息，比如HOG特征，然后利用这些特征去实现定位和分类。而在深度学习这一块，这一任务就交由backbone网络来完成。
- HOG特征：在一副图像中，局部目标的表象和形状能够被梯度或边缘的方向密度分布很好地描述。其本质为：梯度的统计信息，而梯度主要存在于边缘的地方。
- **VGG-16**：
1.**VGG**网络：《**Very Deep Convolutional Networks for Large-Scale Image Recognition》。其中最常用的就是VGG-16.**

2.**ResNet**网络：《**Deep Residual Learning for Image Recognition**》。其中最常用的就是**ResNet50**和**ResNet101**。当任务需求很小的时候，也可以用ResNet18.

3.**ResNeXT**网络：《**Aggregated residual transformations for deep neural networks**》，这个我没有用过，但很多sota工作中都会使用，刷榜的小伙伴不妨考虑一下。

4.**ResNet+DCN**网络：这一网络主要是将DCN工作应用在ResNet网络上，DCN来源于这篇文章：《**Deformable Convolutional Networks**》。DCN是常用的涨点神器，不过似乎在实际部署的时候要复杂一些，刷榜的时候还是很值得一用。

5.**DarkNet网络**：常用的包括**darknet19**和**darknet53**，这两个网络分别来源于YOLOv2和YOLOv3两个工作中。其中darknet19对标的是vgg19，darknet53对标的是resnet101，但由于darknet本身是个很小众的深度学习框架，不受学术界关注，且这两个网络均是由darknet框架实现的，因此也就很少会在其他工作中看到这两个backbone。不过，笔者更偏爱darknet，也对其进行了复现，因为结构简洁，便于理解。

6.**CSPResNet网络**：出自于《**CSPNet: A New Backbone that can Enhance Learning Capability of CNN**》。CSP是一种很好用的结构，在减少参数量的同时，还能够提升模型性能，是不可多得的性价比极高的模块之一。像前一段时间的Scaled-YOLOv4就借鉴了这一工作的思想大幅度提升了YOLOv4的性能。不过，目前似乎也不是主流，仍旧无法撼动ResNet101和ResNet+DCN的刷榜地位。

**然后是轻量型网络：**

1.**MobileNet**：谷歌的工作，一共出了v1，v2，v3三个版本了，相较于上面那些以GPU为主要应用平台的大型网络，MobileNet则着眼于低性能的移动端平台，如手机、嵌入式设备等。

2.**ShuffleNet**：旷视的工作，一共出了v1和v2两个版本，同样是针对于低性能的移动端平台。
##### 2.Neck：更好地利用网络所提取的特征信息
由于backbone网络毕竟是从图像分类（image classification）任务迁移过来的，其提取特征的模式可能不太适合与detection。因此，在我们最终从这些特征中得到图像中若干目标的类别信息（classification）和位置（location）信息之前，有必要对它们做一些处理。
![[Pasted image 20230306151744.png]]
在CNN中，有一个很关键的概念叫做“感受野”（receptive field），**这一张特征图的pixel能包含原始图像中的少个像素**。直观上来看，backbone最后输出的很粗糙的特征图——通常都是stride=32，即经过了32倍降采样——具有很大的感受野，这对于大物体来说是很友好的，但对于小物体而言，过大的感受野且不说容易“失焦”，经过多次降采样，小物体的信息也很容易被丢失掉了。
为了解决这么个问题，SSD在三个不同大小的特征图上进行预测，即上图中的（c），但CNN随着网络深度的增加，每一层的特征图所携带的信息量和信息性质也不一样——浅层包含的细节信息、轮廓信息、位置信息等更多，深层包含的语义信息更多。因此，FPN的工作就是在检测前，先将多个尺度的特征图进行一次bottom-up的融合，也就是上图中的（d），这被证明是极其有效的特征融合方式，几乎成为了后来目标检测的标准模式之一。

除了FPN，还有SPP模块，这也是很常用的一个Neck结构，下图便是SPP的结构示意图。
![[Pasted image 20230306151815.png]]
SPP的思想很简单，通过不同大小的maxpooling核来丰富特征图的感受野。

除此之外，还有：

1.  **RFB**：出自《**Receptive Field Block Net for Accurate and Fast Object Detection**》

2. **ASPP**：出自《**DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs**》

3. **SAM**：出自《**CBAM: Convolutional block attention module**》

4. **PAN**：出自《**Path aggregation network for instance segmentation**》。PAN是一个非常好用的特征融合方式，在FPN的bottom-up基础上又引入了top-down二次融合，有效地提升了模型性能。

##### 3.Detection head：负责检测与定位。
一张图像，在经过了backbone和neck两部分的处理后，就可以准备进行最终的检测了。

随后，在这样的特征图上，通过添加几层卷积即可进行识别和定位。

Detection head通常就是普通的卷积，如下图的RetinaNet：
![[Pasted image 20230306152106.png]]
RetinaNet最后的detection head部分就是三条并行的分支，每个分支右4层普通卷积堆叠而成。

## 1、YOLOv1的网络架构
YOLO-v1最大的特点就在于：**仅使用一个卷积神经网络端到端地实现检测物体的目的**。其网络整体的结构如下图所示：
![[Pasted image 20230306152438.png]]
## 2.YOLOv1的检测原理
一张图像输入给网络，网络最后输出一个7×7×30的特征图。其中， 7×7 是原图 448×448 经过64倍降采样（即网络最终的stride为64）得到的，而通道数**30**的含义是：

特征图的每个位置 预测**两个bounding box（bbox），而每个bbox包含五个输出参数：置信度** � ，**矩形框参数** (cx,cy,w,h) ，**共10个参数**，**再加上20个类别，一共就是30了**。置信度C的作用是判断此处是否有目标的中心点。

通常，置信度C也被记作**objectness预测**，表征此处是否有物体的中心点，即是否有物体。

PS：之所以是20个类别，是因为那时候的数据集只有PASCAL VOC，共20个类别，COCO还没提出来。
更一般的，我们可以用下面的公式来计算这个特征图的通道数：

5B+C
其中， 5 是指边界框的置信度和位置参数； B是每个位置预测的bbox数量， C是类别的数量（如PASCAL VOC中有20个类别，MSCOCO中常用的是80个类别）。
首先，网络的输入是 448×448 的图片，经过网络64倍的降采样后，最后的卷积输出是 7×7 的（ 448÷64=7 ）——在这里停一下，因为这里就体现了YOLOv1的核心思想，也是自此之后绝大部分的one-stage检测的核心范式：

**逐网格找东西**。

具体来说就是，这个 7×7 相当于把原来的448×448 的图片进行了7×7等分，如下图所示：
![[Pasted image 20230306195657.png]]
YOLOv1是想通过看这些网格来找到物体的**中心点坐标**，并确定其**类别。**具体来说，就是每一个网格都会输出 � 个bbox和� 个类别的置信度，而每个bbox包含5个参数（框的置信度+框的坐标参数），因此，每个网格都会给出5�+�个预测参数。因此，网络最终输出的预测参数总量就是：

S * S * ( 5B+C )

其中S = 输入图像尺寸/网络的最大stride， 输入图像尺寸= 448，网络的最大stride=64 ，

总的来说，YOLOv1一共有三部分输出，分别是objectness、class以及bbox：
-   objectness就是上面所说的**框的置信度**，用于表征该网格是否有物体；
-   class就是**类别预测**；
-   而bbox就是**边界框(bounding box)**。

## 3、YOLOv1的正样本制作方法
由于YOLOv1是去预测物体的中心点，并给出矩形框，因此，包含中心点的网格，我们认为这里是有物体的，即这一网格的objectness的概率为1：Pr(Objectness)=1 ，如下图所示：
![[Pasted image 20230306200249.png]]

黄颜色代表这个网格有物体，Pr(Objectness)=1 也就意味着，物体的中心点落在了这个网格中，那么，这个网格就会被标记为一个“**正样本候选区域**”，即**这个标签的正样本只会来源于这个网格**。

_YOLO一共有三个预测：objectness、class、bbox。_
**bjectness是一个二分类，即有物体还是无物体，也就是“边界框的置信度”**，对应loss函数中的那个“C”，**没物体的标签显然就是0**，**而有物体的标签可以直接给1，也可以计算当前预测的bbox与gt之间的IoU作为有物体的标签**，注意，这个IoU是objectness预测学习的标签。
class就是类别预测，只有**正样本候选区域**中的预测框才有可能会被训练，也就是Pr(objectness)=1的地方，注意，这个Pr(objectness)=1就是指正样本候选区域，和IoU没关，和YOLO没关，只和label有关，因为gt box的中心点落在哪个grid，哪个grid就是正样本候选区域，也就是Pr(objectness)=1。正样本候选区域的作用就是告知我们：**该标签的正样本只会来源于此**。
   
## 误差计算：
![[Pasted image 20230307184918.png]]

位置误差：  
xywh与最终真实值之间的误差，S表示S * S的网格，B表示B个预测框，第一种和第二种经验框 ，
wh的误差计算是为了解决一些  数值较小时的影响，加根号后 当x较小时y的变化范围比较明显，x较大是y的变化范围小
i表示每个格子，j表示每个格子里面有两种bandingbox框、
基于置信度预测当前预测宽是前景还是背景

置信度误差含有Object的 (前景 要检测的物体)置信度和真实值之间的误差、
置信度误差不含Object多了权重参数 ，不含物体的 背景，图片中除要检测的物体大部分未背景图片，所以要加入参数弱化背景影响
分类误差：
预测概率-真实概率
NMS（非极大值抑制）
YOLO-V1的缺点，一个中心只能预测一个物体，当两个物体重合时无法精准识别 

##### 目标检测：
imput 预处理 数据增强 归一化 集合光学变化
backBone: 提特征
Neck： 特征融合
head： 输出
loss： 查看结果 

  yolov3适合小目标检测

融入多持续特征图信息预测不同规格的物体
先验框更加丰富，三种scale每种三个规格一共九种
softmax改进