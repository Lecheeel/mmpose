import cv2
import torch
import multiprocessing as mp
import numpy as np
import os
import argparse
import sys

# 修复导入问题
try:
    # 作为包的一部分导入时使用相对导入
    from . import config
    from .utils import SharedData
except ImportError:
    # 直接运行脚本时使用绝对导入
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from multi_camera_pose_estimation import config
    from multi_camera_pose_estimation.utils import SharedData

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多摄像头姿态估计系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--device', type=str, help='设备名称')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
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
            print("调试模式已开启")
            
        # 检测可用设备
        device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建多进程管理器
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # 创建共享数据结构
        shared_data = SharedData()
        shared_data.running.value = True
        
        # 修复导入问题
        try:
            # 作为包的一部分导入时
            from .multiprocess_camera import camera_process
        except ImportError:
            # 直接运行脚本时
            from multi_camera_pose_estimation.multiprocess_camera import camera_process
        
        # 获取模型名称
        model_name = args.model if args.model else config.ModelConfig.DEFAULT_MODEL
        
        # 创建两个摄像头处理进程
        process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, model_name, device))
        process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, model_name, device))
        
        # 启动进程
        process1.start()
        process2.start()
        
        print("按'q'键退出")
        
        while True:
            # 检查是否有帧需要显示
            frames_to_show = False
            
            if 'frame_0' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_0'], dtype=np.uint8)
                display_frame1 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('CAM0 - RTMPose', display_frame1)
                frames_to_show = True
                
            if 'frame_1' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                display_frame2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('CAM1 - RTMPose', display_frame2)
                frames_to_show = True
                
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
        
    finally:
        # 设置退出标志
        shared_data.running.value = False
        
        # 等待进程结束
        if 'process1' in locals() and process1.is_alive():
            process1.join(timeout=1.0)
            if process1.is_alive():
                process1.terminate()
                
        if 'process2' in locals() and process2.is_alive():
            process2.join(timeout=1.0)
            if process2.is_alive():
                process2.terminate()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 