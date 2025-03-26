import yaml
import os
from typing import Dict, Any

# 全局变量控制程序退出
RUNNING = True

class SystemConfig:
    """系统整体配置"""
    
    # 程序控制
    SHOW_DETECTION_COUNT = True     # 是否在控制台显示检测到的人数
    DEBUG_MODE = False              # 是否开启调试模式
    LOG_LEVEL = "INFO"              # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL

class ModelConfig:
    """模型相关配置"""
    
    # 模型选择
    DEFAULT_MODEL = 'rtmpose-l_8xb32-270e_coco-wholebody-384x288'  # 默认姿态估计模型
    
    # 模型量化设置
    USE_FP16 = True                 # 是否使用半精度浮点数(FP16)，提高推理速度
    USE_INFERENCE_MODE = True       # 使用inference_mode而非no_grad
    TENSOR_CORES_ENABLED = True     # 是否启用Tensor Cores (适用于RTX系列显卡)
    
    # 模型预热
    MODEL_WARMUP_COUNT = 10         # 模型预热推理次数

class InferenceConfig:
    """推理配置"""
    
    # 默认推理配置
    DEFAULT_INFERENCE_CONFIG = {
        'show': False,              # 是否显示结果（我们自己处理显示）
        'draw_bbox': True,          # 是否绘制边界框
        'radius': 5,                # 关键点半径
        'thickness': 2,             # 线条粗细
        'kpt_thr': 0.4,             # 关键点置信度阈值
        'bbox_thr': 0.3,            # 边界框置信度阈值
        'nms_thr': 0.65,            # 非极大值抑制阈值，对于rtmpose，更高的NMS阈值通常更好
        'pose_based_nms': True,     # 启用基于姿态的NMS
        'max_num_bboxes': 15        # 最大检测人数上限
    }
    
    # 队列设置
    INPUT_QUEUE_SIZE = 5            # 输入帧队列大小
    OUTPUT_QUEUE_SIZE = 5           # 输出结果队列大小
    
    # 性能统计设置
    STATS_WINDOW_SIZE = 30          # 统计窗口大小（保留最近多少帧的统计数据）
    FPS_RESET_INTERVAL = 100        # 每隔多少帧重置FPS计数器

class TrackingConfig:
    """跟踪配置"""
    
    # 跟踪参数
    TRACKING_THRESHOLD = 0.3        # IOU跟踪阈值
    
    # 缓存清理
    CUDA_CACHE_CLEAR_INTERVAL = 100 # 每隔多少帧清理一次CUDA缓存

class CameraConfig:
    """相机配置"""
    
    # 相机设置
    DEFAULT_CAMERA_WIDTH = 640      # 默认相机宽度
    DEFAULT_CAMERA_HEIGHT = 480     # 默认相机高度
    DEFAULT_CAMERA_FPS = 30         # 默认相机帧率
    CAMERA_RECONNECT_DELAY = 1.0    # 相机重连延迟（秒）
    
    # 图像压缩
    JPEG_QUALITY = 75               # JPEG图像质量（1-100），用于进程间传输图像

class DisplayConfig:
    """显示配置"""
    
    # 文本显示
    INFO_TEXT_POSITION = (10, 30)   # 信息文本位置
    INFO_TEXT_SCALE = 0.7           # 信息文本大小
    INFO_TEXT_COLOR = (0, 255, 0)   # 信息文本颜色 (BGR)
    INFO_TEXT_THICKNESS = 2         # 信息文本粗细
    
    # ID显示
    ID_TEXT_OFFSET = (-18, 0)       # ID文本偏移
    ID_TEXT_SCALE = 0.5             # ID文本大小
    ID_TEXT_COLOR = (255, 255, 255) # ID文本颜色 (BGR)
    ID_TEXT_THICKNESS = 1           # ID文本粗细
    ID_BACKGROUND_PADDING = (20, 15)# ID背景填充大小

class ColorConfig:
    """颜色配置"""
    
    # 骨架连接颜色
    SKELETON_COLORS = [
        (255, 0, 0),   # 鼻子到左眼 - 红色
        (255, 0, 0),   # 鼻子到右眼 - 红色
        (255, 0, 0),   # 左眼到左耳 - 红色
        (255, 0, 0),   # 右眼到右耳 - 红色
        (255, 165, 0), # 颈部连接 - 橙色
        (255, 165, 0), # 颈部连接 - 橙色
        (255, 165, 0), # 颈部连接 - 橙色
        (0, 255, 0),   # 左上肢 - 绿色
        (0, 255, 0),   # 左上肢 - 绿色
        (0, 0, 255),   # 右上肢 - 蓝色
        (0, 0, 255),   # 右上肢 - 蓝色
        (255, 255, 0), # 左肩到左髋 - 黄色
        (255, 0, 255), # 右肩到右髋 - 紫色
        (128, 128, 0), # 髋部连接 - 橄榄色
        (255, 255, 0), # 左下肢 - 黄色
        (255, 255, 0), # 左下肢 - 黄色
        (255, 0, 255), # 右下肢 - 紫色
        (255, 0, 255), # 右下肢 - 紫色
        (0, 255, 255), # 左脚 - 青色
        (0, 255, 255), # 左脚 - 青色
        (128, 0, 128), # 右脚 - 深紫色
        (128, 0, 128)  # 右脚 - 深紫色
    ]
    
    # 关键点颜色
    KEYPOINT_COLOR = (0, 255, 0)     # 关键点颜色
    NECK_VERTEBRA_COLOR = (0, 165, 255) # 颈椎中点颜色
    HIP_MID_COLOR = (165, 0, 255)    # 髋部中点颜色
    SPINE_COLOR = (255, 255, 255)     # 脊柱线颜色
    
    # 边界框颜色
    BBOX_COLOR = (0, 255, 0)          # 边界框颜色
    BBOX_THICKNESS = 2                # 边界框线条粗细
    BBOX_ID_COLOR = (0, 255, 255)     # 边界框ID文本颜色

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(config_file: str = None) -> Dict[str, Any]:
        """从配置文件加载配置
        
        Args:
            config_file: 配置文件路径，如果未提供则使用默认配置
            
        Returns:
            配置字典
        """
        config = {
            'system': {attr: getattr(SystemConfig, attr) for attr in dir(SystemConfig) if not attr.startswith('_')},
            'model': {attr: getattr(ModelConfig, attr) for attr in dir(ModelConfig) if not attr.startswith('_')},
            'inference': {attr: getattr(InferenceConfig, attr) for attr in dir(InferenceConfig) if not attr.startswith('_')},
            'tracking': {attr: getattr(TrackingConfig, attr) for attr in dir(TrackingConfig) if not attr.startswith('_')},
            'camera': {attr: getattr(CameraConfig, attr) for attr in dir(CameraConfig) if not attr.startswith('_')},
            'display': {attr: getattr(DisplayConfig, attr) for attr in dir(DisplayConfig) if not attr.startswith('_')},
            'color': {attr: getattr(ColorConfig, attr) for attr in dir(ColorConfig) if not attr.startswith('_')},
        }
        
        # 如果提供了配置文件，则加载并更新配置
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    
                # 递归更新配置
                def update_config(config, user_config):
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in config:
                            update_config(config[key], value)
                        else:
                            config[key] = value
                
                update_config(config, user_config)
                print(f"已从 {config_file} 加载配置")
            except Exception as e:
                print(f"加载配置文件出错: {str(e)}")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_file: str) -> bool:
        """保存配置到文件
        
        Args:
            config: 配置字典
            config_file: 配置文件路径
            
        Returns:
            是否保存成功
        """
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"已保存配置到 {config_file}")
            return True
        except Exception as e:
            print(f"保存配置文件出错: {str(e)}")
            return False

# 创建默认配置
config = ConfigManager.load_config() 