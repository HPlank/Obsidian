# 安装策略

```bash
# 2.28.2	mmcv-full>=1.3.17, <1.8.0
# 创建虚拟环境
conda create --name openmmlab python==3.9
source activate openmmlab
conda remove -n openmmlab --all  #删除

conda create --name openmmlab_LVPIAN python==3.9
source activate openmmlab_LVPIAN

# 创建cuda环境
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

pip install mmcv-full==1.4 -f http://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

# 从源码安装mmdet：
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。

#	安装额外的依赖以使用 Instaboost, 全景分割, 或者 LVIS 数据集

# 安装 instaboost 依赖
pip install instaboostfast

# 安装全景分割依赖
# https://blog.csdn.net/qq_45961101/article/details/130384117
git clone https://github.com/cocodataset/panopticapi.git  
cd panopticapi-master
pip install -e .
# 安装 LVIS 数据集依赖
git clone https://github.com/lvis-dataset/lvis-api.git
cd lvis-api-master
pip install -e .

# 安装 albumentations 依赖
pip install -r requirements/albu.txt
```

## 修改BUG

https://blog.csdn.net/BUCKY_999/article/details/126976363

# 验证安装

为了验证是否正确安装了 MMDetection 和所需的环境，我们可以运行示例的 Python 代码来初始化检测器并推理一个演示图像：

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: 
#http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
inference_detector(model, 'demo/demo.jpg')
```

# 复现



## 1 自定义数据集的格式转化为coco格式

由于coco格式是多数目标检测算法通用的格式，故这里也将自定义的数据集转化为coco格式，格式如下：

```
|--your coco format dataset name

|--annotations

|--train2017

|--val2017

|--test2017
```

在mmdetection目录下新建data文件夹，将该数据集放入新建的data文件夹中。

## 2、修改数据集

### 1.1 修改configs/_base_/datasets/coco_detection.py

```python
# 根据下列 修改 文件夹
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
```

![image-20230607163411948](mmdetection2.28.2复现_img/image-20230607163411948.png)

### 1.2 修改 mmdet/datasets/coco.py

```python
#	修改 CLASS 和 PALETTE

CLASSES = ('air-hole','broken-arc', 'hollow-bead', 'overlap','unfused','bite-edge', 'crack','slag-inclusion')
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),(106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)]
```

设置COCO评价指标中显示每个类别的AP，参考：https://blog.csdn.net/funky_Lelle/article/details/127345468

在mmdetection/mmdet/datasets/coco.py 找到 evaluate() ：

```python
# 将 evaluate()中的classwise设置为True

# 设置显示按照类别的AP数值
classwise = True
```

### 1.3  修改`mmdet/core/evaluation/class_name.py`文件

```python
# 将其改为自定义数据集的类别名称，注意要和annotations文件夹中的类别顺序一致。如果不改的话，最后测试的结果的名称还会是’aeroplane’, ‘bicycle’, ‘bird’, ‘boat’,…

def coco_classes():
    return [
        'bite-edge', 'crack','slag-inclusion','air-hole','broken-arc', 'hollow-bead', 'overlap','unfused'
    ]
```

### 1.4 修改 /tools/dist_train.sh

```python
#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES="3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
```

## 3、修改学习率、epoch、batchsize

### Tip：修改epoch

```python
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=273)
```

### Tip：修改batch size设置

```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
```

### Tip：修改学习率

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
```

### Tip：学习率计算

> 重要：配置文件中的默认学习率（lr=0.02）是8个GPU和samples_per_gpu=2（批大小= 8 * 2 = 16）。
>
> 根据线性缩放规则，如果您使用不同的GPU或每个GPU的有多少张图像，则需要按批大小设置学习率，例如，
>
> 对于4GPU* 2 img / gpu=8，lr =8/16 * 0.02 = 0.01 ；
>
> 对于16GPU* 4 img / gpu=64，lr =64/16 *0.02 = 0.08 。

**计算公式：lr = (gpu_num × samples_per_gpu) / 16  × 0.02**



## 4、 保存最优权重

```python
configs/_base_/datasets/coco_detection.py

# 将 evaluation = dict(interval=1, metric='bbox')
# 改为
evaluation = dict(interval=1, metric='bbox', save_best='auto')即可。
```

## 5、 mmdetection计算每个类别的AP50

(4条消息) mmdetection计算每个类别的AP50_chao_xy的博客-CSDN博客
https://blog.csdn.net/chao_xy/article/details/130496141

```python
# 修改mmdet/datasets/coco.py的classwise与 iou_thrs
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,#False
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=[0.5],
                 metric_items=None):
```



#  训练

## SSD (ECCV'2016) ✔🍿

### 1 修改类别数

/configs/_base_/models/ssd300.py

```python
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),

        # 修改为类别数
        # num_classes=80,
        num_classes=8,
```

修改学习率

```
optimizer = dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=5e-4)
```

### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/ssd/ssd512_coco.py 1
```

### 3 测试

```python
python tools/train.py configs/xxxx.py --eval mAP

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab
# 测试集结果
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/ssd/ssd512_coco.py work_dirs/ssd512_coco/latest.pth --eval bbox
# 预测结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./work_dirs/ssd512_coco/ssd512_coco.py ./work_dirs/ssd512_coco/latest.pth --show-dir result/SSD-imgs-HANJIE/

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN
# 测试集结果
CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/ssd/ssd512_coco.py work_dirs/ssd512_coco/latest.pth --eval bbox
# 预测结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./work_dirs/ssd512_coco/ssd512_coco.py ./work_dirs/ssd512_coco/latest.pth --show-dir result/SSD-imgs-LVPIAN/
```







## YOLOv3 (ArXiv'2018) 废了

### 1 修改类别数

/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py

```python
    bbox_head=dict(
        type='YOLOV3Head',

        # 修改类别数,
        # num_classes=80,
        num_classes=8,
```

### 2 修改数据集

/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py

```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
```

### 3 运行

应该修改学习率，否则会报错 ERROR - The testing results of the whole dataset is empty.

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

bash ./tools/dist_train.sh ./configs/yolo/yolov3_d53_320_273e_coco.py 1

bash ./tools/dist_train.sh ./configs/ssd/ssd512_coco_HANJIE.py 1

/export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE/configs/ssd/ssd512_coco_HANJIE.py
```



## YOLOF (CVPR'2021) ：❓ 🍿

### 1 修改类别数

/configs/yolof/yolof_r50_c5_8x8_1x_coco.py

```python
    bbox_head=dict(
        type='YOLOFHead',

        # 修改类别
        # num_classes=80,
        num_classes=8,
```



### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py 1
```

### 3 测试

```python
python tools/train.py configs/xxxx.py --eval mAP


cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab
# 预测结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py ./work_dirs/yolof_r50_c5_8x8_iter-1x_coco/latest.pth --show-dir result/YOLOF-imgs-HANJIE/

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN
# 预测结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py ./work_dirs/yolof_r50_c5_8x8_iter-1x_coco/latest.pth --show-dir result/YOLOF-imgs-LVPIAN/




CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py work_dirs/yolof_r50_c5_8x8_iter-1x_coco/iter_12000.pth --eval bbox

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py work_dirs/yolof_r50_c5_8x8_iter-1x_coco/best_bbox_mAP_iter_22500.pth --eval bbox 

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py work_dirs/yolof_r50_c5_8x8_iter-1x_coco/latest.pth --eval bbox




```



## YOLOX (CVPR'2021)：✖ 为0

### 1 修改类别数

/configs/yolox/yolox_s_8x8_300e_coco.py

```
    bbox_head=dict(
        # 修改类别数
        type='YOLOXHead', num_classes=8, in_channels=128, feat_channels=128),
```

### 2 修改数据集

/configs/yolox/yolox_s_8x8_300e_coco.py

```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
```

### 3 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab


cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/yolox/yolox_s_8x8_300e_coco.py 1
```

## DETR (ECCV'2020)  ✖ 为0 多次为0 可以丢弃了

###  1 修改类别数

/configs/detr/detr_r50_8x2_150e_coco.py

```python
    bbox_head=dict(
        type='DETRHead',
        
        # 修改类别数
        # num_classes=80,
        num_classes=8,
```

### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/detr/detr_r50_8x2_150e_coco.py 1
```

===

到这了

===





## TOOD (ICCV'2021)❓ 🍿

### 1 修改类别数

/configs/tood/tood_r50_fpn_1x_coco.py

```python
    bbox_head=dict(
        type='TOODHead',
        # 修改类别
        num_classes=8,
```

### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/tood/tood_r50_fpn_anchor_based_1x_coco.py 1
```

### 3 测试

```python
python tools/train.py configs/xxxx.py --eval mAP


cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab
# 预测结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./configs/tood/tood_r50_fpn_anchor_based_1x_coco.py work_dirs/tood_r50_fpn_anchor_based_1x_coco/latest.pth --show-dir result/TOOD-imgs-HANJIE/


cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN
# 预测结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./configs/tood/tood_r50_fpn_anchor_based_1x_coco.py work_dirs/tood_r50_fpn_anchor_based_1x_coco/latest.pth --show-dir result/TOOD-imgs-LVPIAN/


CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/tood/tood_r50_fpn_anchor_based_1x_coco.py work_dirs/tood_r50_fpn_anchor_based_1x_coco/best_bbox_mAP_epoch_150.pth --eval bbox


CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/tood/tood_r50_fpn_anchor_based_1x_coco.py work_dirs/tood_r50_fpn_anchor_based_1x_coco/latest.pth --eval bbox
```





## DDOD (ACM MM'2021)：✔

### 1 修改类别数

/configs/ddod/ddod_r50_fpn_1x_coco.py

```python
    bbox_head=dict(
        type='DDODHead',

        # 修改类别名
        # num_classes=80,
        num_classes=8,
        
```

### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/ddod/ddod_r50_fpn_1x_coco.py 1



```

### 3 测试

```
python tools/train.py configs/xxxx.py --eval mAP

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/ddod/ddod_r50_fpn_1x_coco.py work_dirs/ddod_r50_fpn_1x_coco/best_bbox_mAP_epoch_91.pth --eval bbox

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/ddod/ddod_r50_fpn_1x_coco.py work_dirs/ddod_r50_fpn_1x_coco/latest.pth --eval bbox

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/ddod/ddod_r50_fpn_1x_coco.py work_dirs/ddod_r50_fpn_1x_coco/latest.pth --eval bbox
```





## Dynamic R-CNN (ECCV'2020) 太复杂了

### 1 修改类别数量

/configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py

```python
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        type='DynamicRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,

            # 修改类别
            num_classes=8,
```

### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

bash ./tools/dist_train.sh ./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py 1
```



## Deformable DETR (ICLR'2021) ❓ 🍿

### 1 修改类别数

/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py

```python
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,

        # 修改类别数
        num_classes=8,
```

### 2 运行

```
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py 1
```

### 3 测试

```python
python tools/train.py configs/xxxx.py --eval mAP

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab
# 结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py work_dirs/deformable_detr_r50_16x2_50e_coco/latest.pth --show-dir result/deformable_detr-imgs-HANJIE/

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN
# 结果可视化
CUDA_VISIBLE_DEVICES=3 python tools/test.py ./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py work_dirs/deformable_detr_r50_16x2_50e_coco/latest.pth --show-dir result/deformable_detr-imgs-LVPIAN/


CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py work_dirs/deformable_detr_r50_16x2_50e_coco/best_bbox_mAP_epoch_124.pth --eval bbox

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py work_dirs/deformable_detr_r50_16x2_50e_coco/latest.pth --eval bbox
```



## Sparse R-CNN (CVPR'2021) ❓

### 1 修改类别数

/configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py

```python
        bbox_head=[
            dict(
                type='DIIHead',

                # 修改类别数
                num_classes=8,
```

### 2 运行

```python
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py 1
```

### 3 测试

```
python tools/train.py configs/xxxx.py --eval mAP

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN


CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py work_dirs/sparse_rcnn_r50_fpn_1x_coco/best_bbox_mAP_epoch_143.pth --eval bbox

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py work_dirs/sparse_rcnn_r50_fpn_1x_coco/latest.pth --eval bbox
```





## VarifocalNet (CVPR'2021) ❓

### 1 修改类别数

/configs/vfnet/vfnet_r50_fpn_1x_coco.py

```python
    bbox_head=dict(
        type='VFNetHead',

        # 修改类别数
        num_classes=8,

```

### 2 运行

```python
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

bash ./tools/dist_train.sh ./configs/vfnet/vfnet_r50_fpn_1x_coco.py 1
```

### 3 测试

```
python tools/train.py configs/xxxx.py --eval mAP

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-LVPIAN
source activate openmmlab_LVPIAN

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/vfnet/vfnet_r50_fpn_1x_coco.py work_dirs/vfnet_r50_fpn_1x_coco/best_bbox_mAP_epoch_82.pth --eval bbox

CUDA_VISIBLE_DEVICES=1 python tools/test.py ./configs/vfnet/vfnet_r50_fpn_1x_coco.py work_dirs/vfnet_r50_fpn_1x_coco/latest.pth --eval bbox
```

## PAA (ECCV'2020) 失败

报错 

ValueError: Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, or increase reg_covar.

### 1 修改类别数

/configs/paa/paa_r50_fpn_1x_coco.py

```python
    bbox_head=dict(
        type='PAAHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,

        # 修改类别数
        # num_classes=80,
        num_classes=8,
```

### 2 运行

```python
cd /export/liguodong/mmdetection-ALL/mmdetection-2.28.2-HANJIE
source activate openmmlab

bash ./tools/dist_train.sh ./configs/paa/paa_r50_fpn_1x_coco.py 1
```

# 绘制混淆矩阵

## 1 生成pkl

```python
python ./tools/test.py ./configs/faster_rcnn_r50_fpn_1x.py ./work_dirs/faster_rcnn_r50_fpn_1x/latest.pth --out=result.pkl
```

## 2 绘制

```python
# !python tools/analysis_tools/confusion_matrix.py -h

!python tools/analysis_tools/confusion_matrix.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_2x_voc_cc.py \
    work_dirs/faster-rcnn_r50_fpn_2x_voc/result_epoch_24.pkl \
    work_dirs/faster-rcnn_r50_fpn_2x_voc \
    --show
```





# 报错

https://github.com/open-mmlab/mmdetection/issues/5152
