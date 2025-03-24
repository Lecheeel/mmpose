# 多摄像头人体姿态估计系统

## 项目简介

这是一个基于MMPose的多摄像头人体姿态估计系统，能够实时捕获和分析多个摄像头中的人体姿态，支持人员跟踪和骨架可视化。本项目由福州理工学院AGELAB开发。

## 功能特点

- 支持多摄像头并行处理
- 实时人体姿态估计
- 人员ID跟踪与标记
- 高性能GPU加速处理
- 骨架和关键点可视化
- 实时FPS和推理时间统计

## 系统要求

- Python 3.7+
- CUDA 9.2+ (推荐CUDA 11.0+)
- PyTorch 1.8+
- 至少两个摄像头设备
- 操作系统：Windows/Linux/macOS

## 项目结构

```
multi_camera_pose_estimation/
├── multi_camera.py         # 多摄像头姿态估计主程序
├── ...
├── ...
.
├── camera_pose_estimation.py  # 单摄像头姿态估计实现
├── one_camera_pose_estimation.py  # 单摄像头测试程序
└── two_cameras.py           # 双摄像头测试程序
```

主要模块说明：
- `FrameProcessor`：处理视频帧的类，支持异步操作
- `CameraCapture`：摄像头捕获类，用于获取视频流
- `SharedData`：共享数据结构，用于多进程间通信
- `camera_process`：每个摄像头独立的处理进程
- `process_pose_results`：姿态结果后处理和可视化

## 安装教程

### 先决条件

**步骤 0.** 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

**步骤 1.** 创建并激活conda环境。

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 安装PyTorch（支持CUDA）

对于GPU平台：
```bash
conda install pytorch torchvision -c pytorch
```
或者指定CUDA版本：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

对于仅CPU平台：
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### 安装依赖库

**步骤 3.** 使用MIM安装MMEngine和MMCV。

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

**步骤 4.** 安装MMPose（两种方式）

方式一：使用pip安装MMPose包（推荐一般用户）
```bash
mim install "mmpose>=1.1.0"
```

方式二：从源代码安装（推荐开发者）
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

**步骤 5.** 安装其他依赖

```bash
pip install opencv-python numpy
```

**步骤 6.** （可选）安装MMDetection用于人体检测

如果需要运行一些依赖MMDetection的演示脚本，可以安装MMDetection：

```bash
mim install "mmdet>=3.1.0"
```

### 验证安装

验证MMPose是否正确安装：

```bash
# 下载配置和检查点文件
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .

# 运行推理演示
python -c "
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # 或 device='cpu'

# 请准备一张有人的图片
results = inference_topdown(model, '测试图片.jpg')
print('推理成功!')
"
```

### 其他安装选项

#### 在Google Colab上安装

```bash
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.1"
!git clone https://github.com/open-mmlab/mmpose.git
%cd mmpose
!pip install -e .
```

#### 使用Docker

```bash
# 构建镜像
docker build -t mmpose docker/

# 运行容器
docker run --gpus all --shm-size=8g -it -v {数据目录}:/mmpose/data mmpose
```

## 使用方法

1. 连接至少两个摄像头到计算机
2. 运行程序:

```bash
python multi_camera_pose_estimation/multi_camera.py
```

3. 按'q'键退出程序

## 自定义配置

可以在`main()`函数中修改以下参数：

- 模型选择：可选择`rtmpose-l_8xb32-270e_coco-wholebody-384x288`或其他MMPose支持的模型
- 摄像头ID：默认使用ID为0和1的两个摄像头
- 关键点阈值：可调整`kpt_thr`参数以控制关键点检测的灵敏度
- 边界框阈值：可调整`bbox_thr`参数以控制人体检测的灵敏度

### 高级配置选项

在`multi_camera.py`文件中，可以配置更多高级选项：

```python
# 在main()函数中
frame_processor_config = {
    'kpt_thr': 0.5,  # 关键点置信度阈值
    'bbox_thr': 0.5,  # 边界框置信度阈值
    'nms_thr': 0.8,   # 非极大值抑制阈值
    'draw_heatmap': False,  # 是否绘制热图
    'show_kpt_idx': False,  # 是否显示关键点索引
    'skeleton_style': 'mmpose',  # 骨架风格
    'line_width': 2,   # 骨架线宽度
    'radius': 4,      # 关键点半径
}

# 修改摄像头分辨率
capture_width = 1280
capture_height = 720
```

## 应用场景

本系统适用于多种实际应用场景：

1. **运动分析**: 跟踪和分析运动员的动作和姿态，提供量化数据
2. **人机交互**: 通过姿态识别实现无接触人机交互界面
3. **健康监测**: 监测老年人或患者的姿态变化，检测异常情况
4. **行为识别**: 识别特定姿态和行为，用于安防或行为分析
5. **动作捕捉**: 低成本的动作捕捉解决方案，用于动画制作
6. **舞蹈教学**: 实时姿态比对，辅助舞蹈教学和练习
7. **工业安全**: 监测工人姿态，预防不良工作姿势导致的职业伤害

## 模型库(ModelZoo)

本项目使用MMPose提供的全身人体姿态估计(Wholebody 2D Keypoint)模型。以下是可用的主要模型系列及其性能特点：

### 1. RTMPose系列模型

RTMPose是一系列实时多人姿态估计模型，针对实时应用场景进行了优化，提供了不同规模和精度的变体：

#### 在COCO-Wholebody数据集上的性能表现

| 模型 | 输入尺寸 | 躯干AP | 足部AP | 面部AP | 手部AP | 全身AP | 模型文件 |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|
| rtmpose-m | 256x192 | 0.680 | 0.619 | 0.842 | 0.516 | 0.606 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth) |
| rtmpose-l | 256x192 | 0.704 | 0.672 | 0.876 | 0.536 | 0.635 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth) |
| rtmpose-l | 384x288 | 0.712 | 0.693 | 0.882 | 0.579 | 0.648 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth) |

### 2. RTMW系列模型

RTMW是基于RTMPose的全身姿态估计高级模型，在Cocktail14多数据集上训练，具有更好的泛化能力：

#### 在Cocktail14数据集上的性能表现

| 模型 | 输入尺寸 | 躯干AP | 足部AP | 面部AP | 手部AP | 全身AP | 模型文件 |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|
| rtmw-m | 256x192 | 0.676 | 0.671 | 0.783 | 0.491 | 0.582 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth) |
| rtmw-l | 256x192 | 0.743 | 0.763 | 0.834 | 0.598 | 0.660 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth) |
| rtmw-x | 256x192 | 0.746 | 0.770 | 0.844 | 0.610 | 0.672 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth) |
| rtmw-l | 384x288 | 0.761 | 0.793 | 0.884 | 0.663 | 0.701 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth) |
| rtmw-x | 384x288 | 0.763 | 0.796 | 0.884 | 0.664 | 0.702 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth) |

### 3. HRNet系列模型

HRNet是经典的高分辨率网络架构，适用于高精度姿态估计任务：

#### 在UBody-COCO-Wholebody数据集上的性能表现

| 模型 | 输入尺寸 | 躯干AP | 足部AP | 面部AP | 手部AP | 全身AP | 模型文件 |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|
| pose_hrnet_w32 | 256x192 | 0.685 | 0.564 | 0.625 | 0.516 | 0.549 | [下载链接](https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/ubody/td-hm_hrnet-w32_8xb64-210e_ubody-coco-256x192-7c227391_20230807.pth) |

### 如何在本项目中使用这些模型

1. 下载所需模型权重文件

```bash
# 示例：下载RTMPose-L模型
mim download mmpose --config rtmpose-l_8xb32-270e_coco-wholebody-384x288 --dest .
```

2. 在`main()`函数中修改模型配置

```python
# 使用RTMPose-L模型示例
process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, 'rtmpose-l_8xb32-270e_coco-wholebody-384x288', device))
process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, 'rtmpose-l_8xb32-270e_coco-wholebody-384x288', device))

# 使用RTMW-X模型示例（更高精度，但可能更慢）
# process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, 'rtmw-x_8xb320-270e_cocktail14-384x288', device))
# process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, 'rtmw-x_8xb320-270e_cocktail14-384x288', device))
```

3. 模型选择建议
   - 对于需要高帧率的实时应用，推荐使用RTMPose-M或RTMPose-L模型
   - 对于需要更高精度的应用，推荐使用RTMW-L或RTMW-X模型
   - 输入尺寸较大的模型(384x288)精度更高，但推理速度稍慢

## 性能优化

- 使用多进程并行处理多个摄像头输入
- CUDA加速和半精度推理(FP16)优化
- 内存复用以减少内存分配开销
- 图像编码质量优化以加快帧传输

### 性能测试数据

在不同硬件环境下的性能表现（使用RTMPose-L模型，输入尺寸384x288）：

| 设备 | GPU | CPU | RAM | 摄像头数量 | 平均FPS | 推理时间(ms) |
|:-----|:---:|:---:|:---:|:----------:|:-------:|:------------:|
| 高端PC | RTX 4090 | i9-13900K | 32GB | 2 | 35-40 | 12-15 |
| 中端PC | RTX 3060 | i7-11700 | 16GB | 2 | 25-30 | 18-25 |
| 低端PC | GTX 1660 | i5-10400 | 8GB | 2 | 15-20 | 30-40 |
| 笔记本 | RTX 3060M | i7-11800H | 16GB | 2 | 20-25 | 25-35 |

## 输出示例

程序运行后会显示两个窗口，分别展示两个摄像头的实时姿态估计结果，包括：

- 检测到的人体骨架
- 每个人的唯一ID标识
- 实时FPS和推理时间统计

## 故障排除

### 常见问题及解决方案

1. **问题**：无法找到摄像头或摄像头打开失败
   **解决方案**：
   - 确认摄像头设备连接正常
   - 检查摄像头设备ID是否正确（默认为0和1）
   - 尝试使用其他视频捕获应用测试摄像头
   - 在Windows系统上，确保摄像头权限已开启

2. **问题**：运行程序时出现"No module named 'mmcv.ops'"错误
   **解决方案**：
   - 确保CUDA版本与PyTorch版本匹配
   - 重新安装mmcv: `pip uninstall mmcv-full mmcv && mim install "mmcv>=2.0.1"`
   - 尝试从源码安装mmcv和mmpose

3. **问题**：显示"CUDA out of memory"错误
   **解决方案**：
   - 尝试使用更小的模型（如RTMPose-M代替RTMPose-L）
   - 减小输入图像尺寸（修改`CameraCapture`类的分辨率参数）
   - 关闭其他GPU密集型应用
   - 在`main()`函数中添加内存管理选项：`torch.cuda.set_per_process_memory_fraction(0.8)`

4. **问题**：跟踪ID不稳定，经常变化
   **解决方案**：
   - 增加IOU跟踪阈值（在`track_by_iou()`函数中）
   - 确保光线充足，减少遮挡
   - 尝试使用更大输入尺寸的模型以提高精度

5. **问题**：推理速度慢
   **解决方案**：
   - 使用更快的模型（RTMPose-M）
   - 减小输入图像尺寸
   - 在GPU支持的情况下，确保使用CUDA加速
   - 关闭不必要的可视化选项

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

- 代码优化和性能改进
- 新特性和功能实现
- 文档改进和翻译
- 错误修复和问题报告

### 贡献流程

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature-name`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature-name`
5. 提交Pull Request

### 代码规范

- 遵循PEP 8 Python编码规范
- 添加适当的类型提示和文档字符串
- 确保代码通过pylint静态检查
- 编写单元测试（如果适用）

## 许可证

本项目采用 Apache 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 参考资料

- [MMPose官方安装文档](https://mmpose.readthedocs.io/en/latest/installation.html)
- [MMPose全身姿态估计模型库](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html)
- [OpenMMLab官方文档](https://openmmlab.com/codebase)
- [COCO Wholebody数据集](https://github.com/jin-s13/COCO-WholeBody)
- [人体姿态估计综述论文](https://arxiv.org/abs/2012.13392)

## 开发团队

本项目由福州理工学院AGELAB开发和维护。

