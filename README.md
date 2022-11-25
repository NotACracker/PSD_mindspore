# Perturbed Self-Distillation: Weakly Supervised Large-Scale Point Cloud Semantic Segmentation (ICCV 2021, Mindspore)

# 论文
- [paper](http://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Perturbed_Self-Distillation_Weakly_Supervised_Large-Scale_Point_Cloud_Semantic_Segmentation_ICCV_2021_paper.pdf)

# 环境配置

- python==3.7.5
- mindspore == 1.7.0

```shell
conda env create -f ./utils/ms170.yaml
bash compile_op.sh
```

# S3DIS数据集准备

## 方式1. 从百度网盘链接下载处理好的数据（推荐）

[百度网盘链接](https://pan.baidu.com/s/101vw5nE-a9CmznWbIcSG_w?pwd=50dh)

数据集目录如下：

```shell
dataset
-- S3DIS # S3DIS数据集
---- input_0.040
------ *.ply
------ *_KDTree.pkl
------ *_proj.pkl
---- original_ply
------ *.ply
PSD_mindspore # PSD代码路径
-- dataset    # 数据集相关的.py文件
---- dataset.py         # 使用gather方式处理点集的dataset文件
---- dataset_mask.py    # 使用mask方式处理点集的dataset文件
-- model   # 模型相关的.py文件
---- model.py         # PSD模型文件
---- loss.py          # loss计算文件,使用gather方式计算loss
---- loss_mask.py     # loss计算文件,使用mask方式计算loss
-- utils   #utils库
-- compile_op.sh
-- psd_s3dis_a5.sh
-- train.py
-- train_mask.py
-- test.py
-- 6_fold_cv.py
```

## 方式2. 从S3DIS官方下载数据集，并执行数据处理

1. S3DIS数据集的链接在 [链接](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1) ，下载`Stanford3dDataset_v1.2_Aligned_Version.zip`，注意不是`Stanford3dDataset_v1.2.zip`
2. 将`Stanford3dDataset_v1.2_Aligned_Version.zip`解压至`dataset/S3DIS`目录下.
3. 安装依赖:

```shell
bash compile_op.sh  #环境配置时运行过此时就不需要运行了
```

4. 数据处理

```shell
python utils/data_prepare_s3dis.py  #设定dataset_path
```

数据集目录格式同方式1

# Ascend环境下训练和验证

## 通过ModelArts的notebook

- 镜像为 `mindspore1.7.0-cann5.1.0-py3.7-euler2.8.3`

### 训练

```shell
python train.py \
--device_target Ascend \
--device_id 0 \
--scale \
--epochs 100 \
--batch_size 4 \
--val_area 5 \
--labeled_point 1% \
--outputs_dir ./outputs \
--name PSD_Area_5_Ascend
```

### 验证

```shell
python test.py \
--device_target Ascend \
--device_id 0 \
--model_path ./outputs/PSD_Area_5_Ascend
```

# GPU环境下训练和验证

### 训练

```shell
python train.py \
--device_target GPU \
--device_id 0 \
--scale \
--epochs 100 \
--batch_size 4 \
--val_area 5 \
--labeled_point 1% \
--outputs_dir ./outputs \
--name PSD_Area_5_GPU
```

### 验证

```shell
python test.py \
--device_target GPU \
--device_id 0 \
--model_path ./outputs/PSD_Area_5_GPU
```

## Citing

### BibTeX

```bibtex
@inproceedings{zhang2021perturbed,
  title={Perturbed self-distillation: Weakly supervised large-scale point cloud semantic segmentation},
  author={Zhang, Yachao and Qu, Yanyun and Xie, Yuan and Li, Zonghao and Zheng, Shanshan and Li, Cuihua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15520--15528},
  year={2021}
}
```
