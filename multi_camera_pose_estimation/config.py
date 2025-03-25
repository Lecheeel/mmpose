# 全局变量控制程序退出
RUNNING = True

# 模型量化设置
USE_FP16 = True  # 是否使用半精度浮点数(FP16)，提高推理速度
USE_INFERENCE_MODE = True  # 使用inference_mode而非no_grad
TENSOR_CORES_ENABLED = True  # 是否启用Tensor Cores (适用于RTX系列显卡)

# 默认推理配置
DEFAULT_INFERENCE_CONFIG = {
    'show': False,  # 我们自己处理显示
    'draw_bbox': False,
    'radius': 5,
    'thickness': 2,
    'kpt_thr': 0.4,
    'bbox_thr': 0.3,
    'nms_thr': 0.65,  # 对于rtmpose，更高的NMS阈值通常更好
    'pose_based_nms': True,  # 启用基于姿态的NMS
    'max_num_bboxes': 15  # 增加检测的最大人数上限
}

# 跟踪配置
TRACKING_THRESHOLD = 0.3  # IOU阈值

# 相机设置
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30 