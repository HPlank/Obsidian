##### IOU指标：
map指标：综合衡量检测效果
![[Pasted image 20230306161856.png]]
 找出女生为正例
 TP：本身是女生判断成女生
 FP: false pre   本身是男生判断成女生
 FN: 本身是女生判断成男生
 TN:本身是男生判断成男生
 精度precision =TP/(TP+FP)

召回率recall=TP/(TP+FN)



yolov2升级

- 加入Batch Normalization
 舍弃Dropout，卷积后加入Batch Normalization，每一次卷积后加入归一化操作，都让特征图中均值为0，方差为thema都对他做归一化.
 每层都做归一化batch Normalization.
- 更大的分辨率
 v1训练时224 * 224 ，测试时448 * 448
 v2训练时额外进行10次448 * 448得微调，map提升%2
- 网络结构
darkNet，实际输入416 * 416
- yolov2聚类提取先验框
-  

yolov3
多scale 为了检测到不同大小的物体，设计了三个scale 