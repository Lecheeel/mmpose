import cv2
import queue
import threading
import time
from typing import Optional, Tuple
import numpy as np
from . import config

class CameraCapture:
    """摄像头捕获类，运行在单独的线程中"""
    
    def __init__(self, camera_id: int = 0, queue_size: int = 5):
        """
        初始化摄像头捕获
        
        Args:
            camera_id: 摄像头ID
            queue_size: 帧队列大小
        """
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.cap = None
        self.frame_count = 0
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
        
    def _capture_worker(self):
        """捕获线程的工作函数"""
        # 创建摄像头对象
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # 使用DirectShow后端，可能提供更好的性能
        
        # 尝试设置更高的捕获分辨率和其他优化
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 设置缓冲区大小为1，减少延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 设置合适的分辨率和帧率，提高效率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DEFAULT_CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DEFAULT_CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.DEFAULT_CAMERA_FPS)
        
        if not self.cap.isOpened():
            print("无法打开摄像头")
            config.RUNNING = False
            return
            
        # 主捕获循环
        while config.RUNNING:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                break
                
            self.frame_count += 1
            
            # 尝试将帧放入队列，如果队列已满则丢弃
            try:
                self.frame_queue.put((self.frame_count, frame), block=False)
            except queue.Full:
                pass  # 队列满时丢弃帧，保持实时性
                
        # 循环结束，释放摄像头
        if self.cap is not None:
            self.cap.release()
            
    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple]:
        """获取一帧视频
        
        Args:
            timeout: 获取超时时间(秒)
            
        Returns:
            (frame_id, frame)元组，或None(如果队列为空)
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release() 