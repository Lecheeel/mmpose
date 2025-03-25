#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义双摄像头姿态估计启动脚本
自动启用自定义关键点可视化模式

使用方法:
    python run_custom_dual_camera.py [--device 设备]

例如:
    python run_custom_dual_camera.py --device cuda:0
"""

import sys
import os

# 将当前目录添加到路径中
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

if __name__ == '__main__':
    import multiprocessing as mp
    from multi_camera_pose_estimation.main import main, parse_args
    import argparse
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='双摄像头姿态估计系统，带自定义关键点可视化')
    parser.add_argument('--device', type=str, help='设备名称，例如cuda:0或cpu')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 构建系统参数
    sys_argv = ['--custom']  # 自动启用自定义关键点可视化模式
    
    if args.device:
        sys_argv.extend(['--device', args.device])
    if args.model:
        sys_argv.extend(['--model', args.model])
    if args.debug:
        sys_argv.append('--debug')
    if args.config:
        sys_argv.extend(['--config', args.config])
    
    # 设置系统参数
    sys.argv = [sys.argv[0]] + sys_argv
    
    # 运行主程序
    main() 