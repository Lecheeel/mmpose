#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义关键点双摄像头主程序
完全独立的实现，不依赖main.py

功能:
    1. 使用双摄像头进行人体姿态估计
    2. 计算并显示自定义解剖学关键点
    3. 实时可视化关键点及其连接线

使用方法:
    python custom_dual_camera_main.py [--device cuda:0] [--model 模型名称] [--debug]

参数:
    --device: 运行设备，例如"cuda:0"(GPU)或"cpu"
    --model: 使用的模型名称，默认为rtmpose-l_8xb32-270e_coco-wholebody-384x288
    --config: 指定配置文件路径
    --debug: 开启调试模式，输出更多信息

自定义关键点: 
    N1: 乳突
    N2: 肩峰
    N3: 第一胸椎
    N4: 第十二胸椎
    N5: 第一腰椎
    N6: 第五腰椎
    ...等17个解剖学关键点

要求:
    - 连接两个摄像头
    - 安装PyTorch、MMPose和OpenCV
"""

import cv2
import torch
import multiprocessing as mp
import numpy as np
import os
import argparse
import sys
import time

# 导入自定义模块
try:
    # 作为包的一部分导入时使用相对导入
    from . import config
    from .utils import SharedData
    from .custom_dual_camera import camera_process_with_custom_viz
except ImportError:
    # 直接运行脚本时使用绝对导入
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from multi_camera_pose_estimation import config
    from multi_camera_pose_estimation.utils import SharedData
    from multi_camera_pose_estimation.custom_dual_camera import camera_process_with_custom_viz

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自定义关键点双摄像头系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--device', type=str, help='设备名称')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--kpt_thr', type=float, default=0.3, help='关键点置信度阈值')
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 如果提供了配置文件，则加载配置
        if args.config and os.path.exists(args.config):
            user_config = config.ConfigManager.load_config(args.config)
            print(f"已加载配置文件: {args.config}")
        
        # 如果命令行指定了调试模式，则更新配置
        if args.debug:
            config.SystemConfig.DEBUG_MODE = True
            print("调试模式已开启，将输出更多信息")
            
            # 额外的调试信息
            print("PyTorch版本:", torch.__version__)
            print("CUDA是否可用:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("CUDA设备数:", torch.cuda.device_count())
                print("当前CUDA设备:", torch.cuda.current_device())
                print("CUDA设备名称:", torch.cuda.get_device_name(0))
            
        # 检测可用设备
        device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建多进程管理器
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # 创建共享数据结构
        shared_data = SharedData()
        shared_data.running.value = True
        
        # 获取模型名称
        model_name = args.model if args.model else config.ModelConfig.DEFAULT_MODEL
        print(f"使用模型: {model_name}")
        
        # 修改配置中的关键点阈值
        if args.kpt_thr:
            config.InferenceConfig.DEFAULT_INFERENCE_CONFIG['kpt_thr'] = args.kpt_thr
            print(f"设置关键点阈值为: {args.kpt_thr}")
        
        # 创建两个摄像头处理进程
        process1 = mp.Process(target=camera_process_with_custom_viz, 
                             args=(0, return_dict, shared_data, model_name, device))
        process2 = mp.Process(target=camera_process_with_custom_viz, 
                             args=(1, return_dict, shared_data, model_name, device))
        
        # 启动进程
        process1.start()
        print(f"摄像头0进程已启动 (PID: {process1.pid})")
        process2.start()
        print(f"摄像头1进程已启动 (PID: {process2.pid})")
        
        print("按'q'键退出")
        
        # 主循环计数器
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            # 检查是否有帧需要显示
            frames_to_show = False
            
            if 'frame_0' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_0'], dtype=np.uint8)
                display_frame1 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('摄像头 0 - 自定义关键点', display_frame1)
                frames_to_show = True
                
            if 'frame_1' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                display_frame2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('摄像头 1 - 自定义关键点', display_frame2)
                frames_to_show = True
            
            # 更新FPS计数
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:  # 每秒更新一次
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
                if config.SystemConfig.DEBUG_MODE:
                    print(f"主线程 FPS: {fps:.1f}")
                
            # 检查错误
            for cam_id in [0, 1]:
                if f'error_{cam_id}' in return_dict:
                    print(f"CAM {cam_id} Error: {return_dict[f'error_{cam_id}']}")
                    return_dict.pop(f'error_{cam_id}', None)
            
            # 检查退出 - 只在有帧显示时处理键盘事件
            if frames_to_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"主进程错误: {str(e)}")
        if config.SystemConfig.DEBUG_MODE:
            import traceback
            traceback.print_exc()
        
    finally:
        # 设置退出标志
        shared_data.running.value = False
        print("正在关闭程序...")
        
        # 等待进程结束
        if 'process1' in locals() and process1.is_alive():
            print("等待摄像头0进程结束...")
            process1.join(timeout=1.0)
            if process1.is_alive():
                print("摄像头0进程未响应，强制终止")
                process1.terminate()
                
        if 'process2' in locals() and process2.is_alive():
            print("等待摄像头1进程结束...")
            process2.join(timeout=1.0)
            if process2.is_alive():
                print("摄像头1进程未响应，强制终止")
                process2.terminate()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 