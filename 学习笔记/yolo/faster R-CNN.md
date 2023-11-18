![[Pasted image 20230625151728.png]]
![[Pasted image 20230625151744.png]]
![[Pasted image 20230625165611.png]].

**图片首先会经过一个预训练过的卷积特征提取层来提取图片的特征**，叫做feature map，
然后feature map通过region Proposal Network网络找出可能包含物体的区域，也就是two-stage的第一阶段
通过RPN之后我们会获得一些可能含有物体的框，再结合第一部分提取的特征图，使用RoI(Region of Interest)将对应物体找出来并把他们的特征提取到新的张量里面进行分类。
最后会通过一个叫做R-CNN的模块将**物体进行分类**，更好地调整框，让边界框更准