# 人体关键点估算与标注工具（摄像头实时版）

此工具可以通过摄像头实时检测人体关键点，并估算额外的解剖学关键点位置，然后将它们标注到视频帧上。

## 功能特点

- 使用RTMPose模型进行实时人体姿态估计
- 基于已检测的关键点，实时估算17个新增解剖学关键点：
  - N1: 乳突
  - N2: 肩峰
  - N3: 第一胸椎
  - N4: 第十二胸椎
  - N5: 第一腰椎
  - N6: 第五腰椎
  - N7: 耻骨联合
  - N8: 髂前上棘
  - N9: 髌骨
  - N10: 第五跖骨
  - N11: 足跟
  - N12: 手掌
  - N13: 脊柱棘突
  - N14: 双侧髂嵴
  - N15: 髂后上嵴
  - N16: 第一跖骨
  - N17: 足尖
- 使用不同颜色标注关键点
- 实时显示处理后的视频帧
- 可选保存处理后的视频帧

## 系统要求

- Python 3.7+
- PyTorch 1.6+
- OpenCV
- MMPose
- 可用的摄像头设备

## 安装

1. 确保安装了Python环境
2. 安装必要的依赖：
   ```
   pip install torch opencv-python
   pip install -U openmim
   mim install mmpose
   ```

## 使用方法

### 命令行运行

```
python keypoint_estimator.py [--camera <摄像头ID>] [--output_dir <输出目录>] [--device <设备>] [--debug] [--save_frames]
```

参数说明：
- `--camera`：摄像头设备ID（默认为0，通常是电脑内置摄像头）
- `--output_dir`：输出图像保存目录（默认为output_images）
- `--device`：计算设备（如cuda:0或cpu，默认为cuda:0）
- `--debug`：启用调试模式，显示更多信息和原始关键点
- `--save_frames`：启用后将保存每一帧处理后的图像到输出目录

### 使用批处理文件运行（Windows）

直接运行提供的批处理文件：
```
run_keypoint_estimator.bat
```

批处理文件会启动摄像头实时处理程序。

### 运行时控制

- 按'q'键可退出程序
- 程序会实时显示检测到的关键点、处理时间和FPS
- 如果启用了`--save_frames`选项，处理后的帧会保存到指定输出目录

## 输出信息

程序会显示以下信息：
- 检测到的关键点数量
- 每帧处理时间
- 实时FPS
- 程序退出时会显示总运行时间、总帧数和平均FPS

## 注意事项

- 首次运行时，程序会自动下载RTMPose模型权重文件
- GPU加速需要安装CUDA和对应版本的PyTorch
- 关键点估算基于检测到的基础人体关键点，因此姿态检测的准确性会影响估算结果
- 对于部分或完全遮挡的身体部位，估算可能不准确
- 处理速度取决于您的电脑性能，GPU通常可以获得更高的FPS 