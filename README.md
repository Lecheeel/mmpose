# 体态与步态评估工具

本工具利用MMPose姿态估计框架，实现体态异常与步态异常分析，可用于跌倒风险评估。工具包含两个主要脚本：

1. `custom_keypoint_visualizer.py` - 用于体态分析，生成和绘制评估所需的关键点
2. `custom_gait_assessment.py` - 用于步态分析，处理视频并计算步态参数

## 功能特点

### 体态分析功能

- 检测并定位17种人体关键点（N1-N17）
- 乳突、肩峰、胸椎、腰椎等关键点的可视化
- 基于关键点的体态评估（头前倾、骨盆倾斜等）

### 步态分析功能

- 步频、步幅、步宽等基本步态参数计算
- 摆动时间、支撑时间、双支撑时间分析
- 足部轨迹可视化
- 步态对称性分析

## 安装要求

1. 安装MMPose和MMDetection：

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
```

2. 下载预训练模型:

```bash
# 人体检测模型
mim download mmdet --config rtmdet_m_640-8xb32_coco-person --dest ./checkpoints

# 姿态估计模型
mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest ./checkpoints
```

## 使用方法

### 体态分析

```bash
python custom_keypoint_visualizer.py --image path/to/image.jpg --save-path output.jpg
```

参数说明:
- `--image`: 输入图像路径
- `--det-config`: 检测模型配置文件
- `--det-checkpoint`: 检测模型权重文件
- `--pose-config`: 姿态估计模型配置文件
- `--pose-checkpoint`: 姿态估计模型权重文件
- `--device`: 推理设备（cuda:0或cpu）
- `--save-path`: 输出图像路径
- `--no-visualize`: 不显示结果

### 步态分析

```bash
python custom_gait_assessment.py --video path/to/video.mp4 --output result.mp4
```

参数说明:
- `--video`: 输入视频路径
- `--det-config`: 检测模型配置文件
- `--det-checkpoint`: 检测模型权重文件
- `--pose-config`: 姿态估计模型配置文件
- `--pose-checkpoint`: 姿态估计模型权重文件
- `--device`: 推理设备（cuda:0或cpu）
- `--output`: 输出视频路径
- `--no-display`: 不显示处理过程

## 评估标准

### 体态评估指标

根据表格评估标准，对以下体态问题进行分析:
- 头前倾（N1、N2）
- 胸脊柱后凸（N3、N4）
- 平背（N5、N6）
- 骨盆后倾/前倾（N7、N8）
- 膝过伸（N9、N10、N11）
- 肩内旋/外旋（N12）
- 脊柱侧弯（N13）
- 骨盆向侧方倾斜（N14）
- 骨盆旋转（N8、N7、N15）
- 足弓异常（N16、N10、N11）
- 膝外翻/内翻（N9、N17）

### 步态评估指标

基于以下参数评估步态异常:
- 步宽（B1：M1-M2横向距离）
- 步长（B2：M1-M2纵向距离）
- 跨步长（B3：M1-M3/M2-M4纵向距离）
- 步频（B4：一分钟内足落地次数）
- 左右两侧步长差异（B5）
- 单腿支撑时间差异（B6、B7）
- 骨盆旋转幅度（B8）
- 骨盆对称性（B9）
- 膝关节屈伸角度（B10）
- 踝关节背屈和跖屈（B11）
- 身体重心转移（B12）
- 左右两侧动作对称性（B13）

## 示例输出

### 体态分析输出

标记的图像中包含以下信息:
- N1-N17关键点位置和标签
- 脊柱连接线
- 关键点间连接关系

### 步态分析输出

处理后的视频和分析报告包含:
- 步数、步宽、步长、步频等基本参数
- 摆动时间、支撑时间、双支撑时间
- 足部轨迹可视化
- 左右对称性分析

