import queue
import threading
import time
import torch
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from mmpose.apis.inferencers import MMPoseInferencer
from . import config
from .utils import torch_inference_mode, init_torch_settings

class FrameProcessor:
    """处理视频帧的类，支持异步操作"""
    
    def __init__(self, model_name: str, device: str, 
                 input_queue_size: int = 5,
                 output_queue_size: int = 5,
                 model_config: Dict = None):
        """
        初始化帧处理器
        
        Args:
            model_name: 要使用的模型名称
            device: 使用的设备('cpu'或'cuda:0'等)
            input_queue_size: 输入队列大小
            output_queue_size: 输出队列大小
            model_config: 模型配置参数
        """
        self.device = device
        self.model_name = model_name
        self.input_queue = queue.Queue(maxsize=input_queue_size)
        self.output_queue = queue.Queue(maxsize=output_queue_size)
        
        # 默认推理配置
        self.call_args = config.DEFAULT_INFERENCE_CONFIG.copy()
        
        # 如果提供了模型配置，则更新配置
        if model_config:
            self.call_args.update(model_config)
            
        # 延迟初始化模型，以便在线程中进行
        self.inferencer = None
        
        # 性能统计
        self.inference_times = []
        self.last_inference_time = 0
        
        # 启动处理线程
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
    def _init_model(self):
        """初始化姿态估计模型"""
        try:
            # 设置CUDNN加速参数
            init_torch_settings(self.device)
                    
            self.inferencer = MMPoseInferencer(
                pose2d=self.model_name,
                device=self.device,
                scope='mmpose',
                show_progress=False
            )
            
            # 执行模型预热，使CUDA初始化所有缓存和进行图优化
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with torch_inference_mode():
                for _ in range(10):  # 多次预热
                    _ = list(self.inferencer(dummy_frame))
                # 强制同步GPU，确保预热完成
                if 'cuda' in self.device:
                    torch.cuda.synchronize()
                    
            print(f"成功加载模型到 {self.device}")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
            
    def _inference_worker(self):
        """推理线程的工作函数"""
        if not self._init_model():
            config.RUNNING = False
            return
            
        # 创建固定的RGB图像缓存，避免重复内存分配
        img_cache = None
        
        while config.RUNNING:
            try:
                # 尝试从队列获取帧，有1秒超时
                try:
                    frame_data = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                frame_id, frame = frame_data
                
                # 开始计时
                start_time = time.time()
                
                # 优化：使用预分配的缓存进行颜色转换，避免重复内存分配
                if img_cache is None or img_cache.shape != frame.shape:
                    img_cache = np.empty(frame.shape, dtype=np.uint8)
                
                # 直接在目标缓冲区上进行颜色转换，避免创建中间数组
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, img_cache)
                
                # 使用torch推理模式提高性能
                with torch_inference_mode():
                    # 使用MMPose进行推理
                    results = list(self.inferencer(img_cache, **self.call_args))
                    # 确保GPU操作完成，避免延迟抖动
                    if 'cuda' in self.device:
                        torch.cuda.synchronize()
                
                # 计算推理时间
                inference_time = time.time() - start_time
                self.last_inference_time = inference_time
                self.inference_times.append(inference_time)
                # 保持统计列表在合理大小
                if len(self.inference_times) > 30:
                    self.inference_times.pop(0)
                
                # 将结果放入输出队列
                self.output_queue.put((frame_id, frame, results, inference_time))
                self.input_queue.task_done()
                
            except Exception as e:
                print(f"推理过程中出错: {str(e)}")
                # 出错时也标记任务完成
                if 'frame_data' in locals():
                    self.input_queue.task_done()
    
    def add_frame(self, frame_id: int, frame: np.ndarray):
        """添加帧到处理队列
        
        Args:
            frame_id: 帧ID
            frame: 视频帧
        """
        try:
            self.input_queue.put((frame_id, frame), block=False)
            return True
        except queue.Full:
            # 如果队列已满，放弃这一帧
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """获取处理结果
        
        Args:
            timeout: 获取超时时间(秒)
            
        Returns:
            (frame_id, frame, results, inference_time)元组，或None(如果队列为空)
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_avg_inference_time(self) -> float:
        """获取平均推理时间"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times) 