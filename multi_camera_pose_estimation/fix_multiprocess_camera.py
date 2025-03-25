import cv2
import torch
import multiprocessing as mp
import numpy as np
import os
import sys
import time
import argparse

# 导入自定义模块
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from multi_camera_pose_estimation import config
from multi_camera_pose_estimation.utils import SharedData
from multi_camera_pose_estimation.custom_dual_camera import compute_custom_keypoints, draw_custom_keypoints

def direct_frame_processor(frame, pose_model, camera_idx, args):
    """最简单的帧处理函数，直接使用模型进行推理并显示结果
    
    Args:
        frame: 输入图像帧
        pose_model: 姿态估计模型
        camera_idx: 摄像头索引
        args: 命令行参数
        
    Returns:
        display_frame: 处理后的显示帧
    """
    display_frame = frame.copy()
    
    try:
        # 使用模型直接推理
        pose_results = pose_model(frame)
        
        # 打印调试信息
        if args.debug:
            print(f"\n[CAM {camera_idx}] 推理结果类型: {type(pose_results)}")
            if hasattr(pose_results, 'get'):
                print(f"[CAM {camera_idx}] 结果键: {pose_results.keys()}")
            elif isinstance(pose_results, list) and pose_results:
                print(f"[CAM {camera_idx}] 结果列表长度: {len(pose_results)}")
                if hasattr(pose_results[0], '__dict__'):
                    print(f"[CAM {camera_idx}] 首个结果属性: {dir(pose_results[0])[:10]}")
        
        # 提取关键点 - 尝试不同的方法
        keypoints = None
        keypoint_scores = None
        
        if isinstance(pose_results, list) and pose_results:
            result = pose_results[0]
            
            # MMPose最新版 - 适应result.pred_instances结构
            if hasattr(result, 'pred_instances'):
                if hasattr(result.pred_instances, 'keypoints') and len(result.pred_instances.keypoints) > 0:
                    keypoints = result.pred_instances.keypoints[0].cpu().numpy()
                    if args.debug:
                        print(f"[CAM {camera_idx}] 使用pred_instances.keypoints获取关键点，形状: {keypoints.shape}")
                
                if hasattr(result.pred_instances, 'keypoint_scores') and len(result.pred_instances.keypoint_scores) > 0:
                    keypoint_scores = result.pred_instances.keypoint_scores[0].cpu().numpy()
                    if args.debug:
                        print(f"[CAM {camera_idx}] 使用pred_instances.keypoint_scores获取分数，形状: {keypoint_scores.shape}")
        
        # 如果没有找到关键点，则显示错误
        if keypoints is None or keypoint_scores is None:
            cv2.putText(display_frame, f"CAM {camera_idx} | 未找到关键点", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if args.debug:
                print(f"[CAM {camera_idx}] 未能提取关键点或分数")
            return display_frame
        
        # 绘制原始关键点
        valid_keypoints = 0
        for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
            if score > args.kpt_thr:
                valid_keypoints += 1
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
                # 显示关键点索引
                cv2.putText(display_frame, str(i), (x + 3, y - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if args.debug:
            print(f"[CAM {camera_idx}] 有效原始关键点数: {valid_keypoints}/{len(keypoints)}")
        
        # 计算自定义关键点
        custom_points = compute_custom_keypoints(keypoints, keypoint_scores, thr=args.kpt_thr)
        
        if args.debug:
            print(f"[CAM {camera_idx}] 计算了{len(custom_points)}个自定义关键点")
            # 打印一些自定义关键点信息
            for key in list(custom_points.keys())[:3]:
                print(f"[CAM {camera_idx}] 关键点 {key}: {custom_points[key]}")
        
        # 绘制自定义关键点
        display_frame = draw_custom_keypoints(display_frame, custom_points, 
                                              radius=10, thickness=4)
        
        # 添加相机信息
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(display_frame, f"CAM {camera_idx} | {timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"有效关键点: {valid_keypoints}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"[CAM {camera_idx}] 处理帧错误: {str(e)}")
        cv2.putText(display_frame, f"CAM {camera_idx} | 错误: {str(e)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if args.debug:
            import traceback
            traceback.print_exc()
    
    return display_frame

def direct_camera_process(camera_idx, args):
    """直接处理摄像头视频的函数，不使用多进程复杂结构
    
    Args:
        camera_idx: 摄像头索引
        args: 命令行参数
    """
    try:
        print(f"[CAM {camera_idx}] 正在初始化...")
        
        # 检查PyTorch和CUDA
        if args.debug:
            print(f"[CAM {camera_idx}] PyTorch版本: {torch.__version__}")
            print(f"[CAM {camera_idx}] CUDA是否可用: {torch.cuda.is_available()}")
        
        # 设置设备
        device = args.device
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"[CAM {camera_idx}] 警告: CUDA不可用，使用CPU")
            device = 'cpu'
        
        # 加载模型
        print(f"[CAM {camera_idx}] 加载模型中...")
        
        # 导入mmpose依赖
        try:
            from mmpose.apis import init_model
            
            if args.debug:
                print(f"[CAM {camera_idx}] 成功导入mmpose")
        except ImportError as e:
            print(f"[CAM {camera_idx}] 导入mmpose失败: {e}")
            return
        
        # 模型配置
        model_config = "configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
        model_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
        
        try:
            pose_model = init_model(model_config, model_checkpoint, device=device)
            print(f"[CAM {camera_idx}] 模型加载成功")
        except Exception as e:
            print(f"[CAM {camera_idx}] 模型加载失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return
        
        # 打开摄像头
        print(f"[CAM {camera_idx}] 打开摄像头...")
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"[CAM {camera_idx}] 无法打开摄像头")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"[CAM {camera_idx}] 初始化完成，开始处理视频")
        
        # FPS计数器
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        
        # 主循环
        while not args.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[CAM {camera_idx}] 无法读取帧，尝试重新初始化摄像头")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(camera_idx)
                continue
            
            # 处理帧
            display_frame = direct_frame_processor(frame, pose_model, camera_idx, args)
            
            # 显示帧速率
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
                if args.debug:
                    print(f"[CAM {camera_idx}] FPS: {fps:.1f}")
            
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示帧
            window_name = f"摄像头 {camera_idx} - 直接处理"
            cv2.imshow(window_name, display_frame)
            
            # 检查退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                args.stop_event.set()
                break
        
        # 释放资源
        cap.release()
        cv2.destroyWindow(window_name)
        print(f"[CAM {camera_idx}] 处理结束")
        
    except Exception as e:
        print(f"[CAM {camera_idx}] 进程错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(description='直接处理摄像头视频的关键点检测')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备，如cuda:0或cpu')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--kpt_thr', type=float, default=0.3, help='关键点置信度阈值')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建停止事件
    args.stop_event = mp.Event()
    
    try:
        # 检查摄像头
        print("正在检查摄像头...")
        cam0 = cv2.VideoCapture(0)
        cam1 = cv2.VideoCapture(1)
        
        available_cameras = []
        if cam0.isOpened():
            available_cameras.append(0)
            cam0.release()
        
        if cam1.isOpened():
            available_cameras.append(1)
            cam1.release()
            
        print(f"可用摄像头: {available_cameras}")
        
        if not available_cameras:
            print("错误: 没有可用的摄像头")
            return
        
        # 创建多进程
        processes = []
        for cam_idx in available_cameras:
            p = mp.Process(target=direct_camera_process, args=(cam_idx, args))
            processes.append(p)
        
        # 启动进程
        for p in processes:
            p.start()
            print(f"进程 {p.pid} 已启动")
        
        print("按'q'键退出")
        
        # 等待用户输入退出
        try:
            while not args.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("接收到退出信号")
            args.stop_event.set()
        
        # 等待进程结束
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                print(f"进程 {p.pid} 未响应，强制终止")
                p.terminate()
        
        print("所有进程已结束")
        
    except Exception as e:
        print(f"主程序错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    finally:
        # 确保清理
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 