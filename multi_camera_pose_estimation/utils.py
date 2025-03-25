import torch
import multiprocessing as mp
from contextlib import contextmanager
from . import config

@contextmanager
def torch_inference_mode():
    """使用torch推理模式的上下文管理器，优化推理性能"""
    if config.USE_INFERENCE_MODE:
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=config.USE_FP16):
            yield
    else:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=config.USE_FP16):
            yield

def init_torch_settings(device):
    """初始化PyTorch设置以优化性能
    
    Args:
        device: 使用的设备('cpu'或'cuda:0'等)
    """
    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        if config.TENSOR_CORES_ENABLED:
            # 启用TensorCores以加速卷积运算
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

# 多进程共享数据的结构
class SharedData:
    """进程间共享数据的结构类"""
    def __init__(self):
        # 创建共享变量用于通信
        self.running = mp.Value('b', True) 