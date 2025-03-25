import os
import sys
import cv2
import numpy as np
import argparse
import time
import torch
import multiprocessing as mp
from collections import deque

# 导入自定义模块
try:
    # 作为包的一部分导入时使用相对导入
    from . import config
    from .utils import SharedData
    from .multiprocess_camera import camera_process
except ImportError:
    # 直接运行脚本时使用绝对导入
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from multi_camera_pose_estimation import config
    from multi_camera_pose_estimation.utils import SharedData
    from multi_camera_pose_estimation.multiprocess_camera import camera_process

class GaitAnalyzer:
    """步态分析器，用于计算步态相关指标"""
    
    def __init__(self, history_size=30):
        """初始化
        
        Args:
            history_size: 历史帧数量
        """
        # 关键点历史记录
        self.history = deque(maxlen=history_size)
        
        # 关键点索引
        self.LEFT_ANKLE_IDX = 15
        self.RIGHT_ANKLE_IDX = 16
        self.LEFT_KNEE_IDX = 13
        self.RIGHT_KNEE_IDX = 14
        self.LEFT_HIP_IDX = 11
        self.RIGHT_HIP_IDX = 12
        
        # 步态周期状态
        self.last_step_time = None
        self.step_count = 0
        self.left_swing_start = None
        self.right_swing_start = None
        self.is_left_swing = False
        self.is_right_swing = False
        self.stride_lengths = []
        self.step_lengths = []
        self.step_widths = []
        self.swing_times = []
        self.stance_times = []
        self.double_support_times = []
        
        # 临时关键点记录
        self.left_foot_positions = []
        self.right_foot_positions = []
    
    def reset(self):
        """重置分析器状态"""
        self.history.clear()
        self.last_step_time = None
        self.step_count = 0
        self.left_swing_start = None
        self.right_swing_start = None
        self.is_left_swing = False
        self.is_right_swing = False
        self.stride_lengths = []
        self.step_lengths = []
        self.step_widths = []
        self.swing_times = []
        self.stance_times = []
        self.double_support_times = []
        self.left_foot_positions = []
        self.right_foot_positions = []
    
    def add_frame(self, keypoints, keypoint_scores, timestamp=None):
        """添加一帧关键点数据
        
        Args:
            keypoints: 关键点坐标数组
            keypoint_scores: 关键点置信度分数
            timestamp: 时间戳，为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 包装数据
        frame_data = {
            'keypoints': keypoints,
            'scores': keypoint_scores,
            'timestamp': timestamp
        }
        
        # 添加到历史记录
        self.history.append(frame_data)
        
        # 分析步态
        self._analyze_gait()
    
    def _is_valid_keypoint(self, frame_data, idx, threshold=0.5):
        """检查关键点是否有效"""
        scores = frame_data['scores']
        return idx < len(scores) and scores[idx] > threshold
    
    def _is_foot_on_ground(self, frame_data, is_left=True):
        """判断脚是否接触地面"""
        ankle_idx = self.LEFT_ANKLE_IDX if is_left else self.RIGHT_ANKLE_IDX
        knee_idx = self.LEFT_KNEE_IDX if is_left else self.RIGHT_KNEE_IDX
        
        # 确保关键点有效
        if not self._is_valid_keypoint(frame_data, ankle_idx) or \
           not self._is_valid_keypoint(frame_data, knee_idx):
            return False
        
        # 获取最低的脚踝位置和平均脚踝高度
        ankles_y = []
        for frame in self.history:
            if self._is_valid_keypoint(frame, self.LEFT_ANKLE_IDX):
                ankles_y.append(frame['keypoints'][self.LEFT_ANKLE_IDX][1])
            if self._is_valid_keypoint(frame, self.RIGHT_ANKLE_IDX):
                ankles_y.append(frame['keypoints'][self.RIGHT_ANKLE_IDX][1])
        
        if not ankles_y:
            return False
        
        # 计算脚踝的阈值高度 (考虑几帧的历史高度)
        max_y = max(ankles_y)
        threshold_y = max_y - 30  # 允许30像素的偏差
        
        # 检查脚踝高度
        ankle_y = frame_data['keypoints'][ankle_idx][1]
        
        # 如果脚踝接近最低高度，认为脚在地面上
        return ankle_y >= threshold_y
    
    def _get_foot_position(self, frame_data, is_left=True):
        """获取脚的位置"""
        ankle_idx = self.LEFT_ANKLE_IDX if is_left else self.RIGHT_ANKLE_IDX
        
        if self._is_valid_keypoint(frame_data, ankle_idx):
            return frame_data['keypoints'][ankle_idx]
        
        return None
    
    def _analyze_gait(self):
        """分析步态周期和参数"""
        if len(self.history) < 2:
            return
        
        current_frame = self.history[-1]
        prev_frame = self.history[-2]
        
        # 获取脚的位置
        left_foot = self._get_foot_position(current_frame, is_left=True)
        right_foot = self._get_foot_position(current_frame, is_left=False)
        
        if left_foot is not None:
            self.left_foot_positions.append(left_foot)
        if right_foot is not None:
            self.right_foot_positions.append(right_foot)
        
        # 限制记录的脚位置数量
        max_positions = 30
        if len(self.left_foot_positions) > max_positions:
            self.left_foot_positions = self.left_foot_positions[-max_positions:]
        if len(self.right_foot_positions) > max_positions:
            self.right_foot_positions = self.right_foot_positions[-max_positions:]
        
        # 检测步态状态
        current_left_on_ground = self._is_foot_on_ground(current_frame, is_left=True)
        current_right_on_ground = self._is_foot_on_ground(current_frame, is_left=False)
        prev_left_on_ground = self._is_foot_on_ground(prev_frame, is_left=True)
        prev_right_on_ground = self._is_foot_on_ground(prev_frame, is_left=False)
        
        # 检测左脚摆动开始
        if prev_left_on_ground and not current_left_on_ground and not self.is_left_swing:
            self.is_left_swing = True
            self.left_swing_start = current_frame['timestamp']
        
        # 检测左脚摆动结束
        if not prev_left_on_ground and current_left_on_ground and self.is_left_swing:
            self.is_left_swing = False
            if self.left_swing_start is not None:
                swing_time = current_frame['timestamp'] - self.left_swing_start
                self.swing_times.append(swing_time)
                self.left_swing_start = None
        
        # 检测右脚摆动开始
        if prev_right_on_ground and not current_right_on_ground and not self.is_right_swing:
            self.is_right_swing = True
            self.right_swing_start = current_frame['timestamp']
        
        # 检测右脚摆动结束
        if not prev_right_on_ground and current_right_on_ground and self.is_right_swing:
            self.is_right_swing = False
            if self.right_swing_start is not None:
                swing_time = current_frame['timestamp'] - self.right_swing_start
                self.swing_times.append(swing_time)
                self.right_swing_start = None
        
        # 计算步长和步宽
        if current_left_on_ground and current_right_on_ground and \
           self.left_foot_positions and self.right_foot_positions:
            # 计算步宽 (足左右间距)
            left_foot = self.left_foot_positions[-1]
            right_foot = self.right_foot_positions[-1]
            step_width = abs(left_foot[0] - right_foot[0])
            self.step_widths.append(step_width)
            
            # 如果有前后位移，计算步长
            if len(self.left_foot_positions) > 1 and len(self.right_foot_positions) > 1:
                prev_left_foot = self.left_foot_positions[-2]
                prev_right_foot = self.right_foot_positions[-2]
                
                # 左脚前进的步长
                left_step_length = abs(left_foot[1] - prev_left_foot[1])
                # 右脚前进的步长
                right_step_length = abs(right_foot[1] - prev_right_foot[1])
                
                # 添加有效的步长
                if left_step_length > 10:  # 忽略很小的变化
                    self.step_lengths.append(left_step_length)
                if right_step_length > 10:
                    self.step_lengths.append(right_step_length)
        
        # 检测步态事件并计数
        if (prev_left_on_ground and not current_left_on_ground) or \
           (prev_right_on_ground and not current_right_on_ground):
            # 脚离开地面，计一步
            self.step_count += 1
            
            # 记录步时
            current_time = current_frame['timestamp']
            if self.last_step_time is not None:
                step_time = current_time - self.last_step_time
                if step_time > 0.1 and step_time < 2.0:  # 忽略不合理的步时
                    self.stance_times.append(step_time)
            self.last_step_time = current_time
        
        # 检测双支撑时间
        if current_left_on_ground and current_right_on_ground:
            # 双脚支撑
            if not (prev_left_on_ground and prev_right_on_ground):
                # 双支撑刚开始
                self.double_support_start = current_frame['timestamp']
        elif prev_left_on_ground and prev_right_on_ground:
            # 双支撑刚结束
            if hasattr(self, 'double_support_start') and self.double_support_start is not None:
                double_support_time = current_frame['timestamp'] - self.double_support_start
                self.double_support_times.append(double_support_time)
                self.double_support_start = None
    
    def get_results(self):
        """获取步态分析结果"""
        results = {
            'step_count': self.step_count,
            'step_width': np.mean(self.step_widths) if self.step_widths else 0,
            'step_length': np.mean(self.step_lengths) if self.step_lengths else 0,
            'stride_length': np.mean(self.stride_lengths) if self.stride_lengths else 0,
            'swing_time': np.mean(self.swing_times) if self.swing_times else 0,
            'stance_time': np.mean(self.stance_times) if self.stance_times else 0,
            'double_support_time': np.mean(self.double_support_times) if self.double_support_times else 0
        }
        
        # 计算步频 (步数/总时间)
        if len(self.history) >= 2:
            total_time = self.history[-1]['timestamp'] - self.history[0]['timestamp']
            if total_time > 0:
                results['step_frequency'] = self.step_count / total_time * 60  # 步/分钟
            else:
                results['step_frequency'] = 0
        else:
            results['step_frequency'] = 0
        
        return results
    
    def draw_foot_trajectory(self, image):
        """在图像上绘制脚的轨迹
        
        Args:
            image: 输入图像
        
        Returns:
            image: 绘制了脚轨迹的图像
        """
        result_image = image.copy()
        
        # 绘制左脚轨迹
        for i in range(1, len(self.left_foot_positions)):
            pt1 = (int(self.left_foot_positions[i-1][0]), int(self.left_foot_positions[i-1][1]))
            pt2 = (int(self.left_foot_positions[i][0]), int(self.left_foot_positions[i][1]))
            cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)
        
        # 绘制右脚轨迹
        for i in range(1, len(self.right_foot_positions)):
            pt1 = (int(self.right_foot_positions[i-1][0]), int(self.right_foot_positions[i-1][1]))
            pt2 = (int(self.right_foot_positions[i][0]), int(self.right_foot_positions[i][1]))
            cv2.line(result_image, pt1, pt2, (255, 0, 0), 2)
        
        return result_image

def draw_gait_metrics(image, gait_results):
    """在图像上绘制步态指标
    
    Args:
        image: 输入图像
        gait_results: 步态分析结果
    
    Returns:
        image: 绘制了步态指标的图像
    """
    result_image = image.copy()
    
    # 设置文本区域
    text_x = 10
    text_y = 30
    line_height = 25
    
    # 绘制指标
    metrics = [
        f"步数: {gait_results['step_count']}",
        f"步宽: {gait_results['step_width']:.2f} 像素",
        f"步长: {gait_results['step_length']:.2f} 像素",
        f"步频: {gait_results['step_frequency']:.2f} 步/分钟",
        f"摆动时间: {gait_results['swing_time']*1000:.2f} 毫秒",
        f"支撑时间: {gait_results['stance_time']*1000:.2f} 毫秒",
        f"双支撑时间: {gait_results['double_support_time']*1000:.2f} 毫秒"
    ]
    
    # 添加半透明背景
    overlay = result_image.copy()
    cv2.rectangle(overlay, (text_x-5, text_y-25), (text_x+300, text_y+line_height*len(metrics)), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
    
    # 绘制文本
    for i, text in enumerate(metrics):
        cv2.putText(result_image, text, (text_x, text_y + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result_image

def process_keypoints_for_gait(results, camera_idx):
    """从姿态估计结果中提取关键点数据
    
    Args:
        results: 姿态估计结果
        camera_idx: 摄像头索引
        
    Returns:
        keypoints, scores: 关键点坐标和置信度分数
    """
    # 默认返回空值
    empty_keypoints = np.zeros((17, 2))
    empty_scores = np.zeros(17)
    
    if not results:
        return empty_keypoints, empty_scores
    
    try:
        # 处理MMPose 2.x返回的结果结构
        for result in results:
            pred_instances = result.get('predictions', [])
            
            if pred_instances and len(pred_instances) > 0:
                # 处理第一个预测列表（通常是姿态估计结果）
                for instance_list in pred_instances:
                    if isinstance(instance_list, list) and len(instance_list) > 0:
                        # 使用第一个检测到的人（假设是主要目标）
                        instance = instance_list[0]
                        
                        # 获取关键点和分数
                        keypoints = instance.get('keypoints', None)
                        kpt_scores = instance.get('keypoint_scores', None)
                        
                        if keypoints is not None and kpt_scores is not None:
                            return np.array(keypoints), np.array(kpt_scores)
    except Exception as e:
        print(f"处理关键点时出错: {str(e)}")
    
    return empty_keypoints, empty_scores

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双摄像头步态分析系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--device', type=str, help='设备名称')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径')
    parser.add_argument('--no-display', action='store_true', help='不显示处理过程')
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
        
        # 获取模型名称
        model_name = args.model if args.model else config.ModelConfig.DEFAULT_MODEL
        
        # 只启动一个摄像头进程
        process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, model_name, device))
        # process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, model_name, device))
        
        # 只启动一个进程
        process1.start()
        # process2.start()
        
        # 创建步态分析器
        gait_analyzer = GaitAnalyzer()
        
        # 创建视频写入器
        output_writer = None
        if args.output:
            # 等待摄像头返回第一帧以获取尺寸
            wait_start = time.time()
            while 'frame_0' not in return_dict and 'frame_1' not in return_dict:
                if time.time() - wait_start > 5.0:  # 5秒超时
                    print("等待摄像头超时")
                    break
                time.sleep(0.1)
            
            # 确定输出视频的尺寸
            output_width = config.CameraConfig.DEFAULT_CAMERA_WIDTH * 2  # 两个摄像头并排
            output_height = config.CameraConfig.DEFAULT_CAMERA_HEIGHT
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(args.output, fourcc, 30, (output_width, output_height))
        
        print("按'q'键退出")
        
        # 主循环
        frame_count = 0
        last_time = time.time()
        fps = 0
        
        while shared_data.running.value:
            frame_count += 1
            current_time = time.time()
            
            # 每秒更新一次FPS
            if current_time - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = current_time
            
            # 检查是否有帧需要显示
            frames_to_show = False
            combined_frame = None
            
            # 处理摄像头0的帧
            if 'frame_0' in return_dict and 'results_0' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_0'], dtype=np.uint8)
                display_frame1 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                
                # 处理姿态估计结果
                results = return_dict['results_0']
                keypoints, keypoint_scores = process_keypoints_for_gait(results, 0)
                
                # 更新步态分析器
                if np.any(keypoint_scores > 0.5):
                    gait_analyzer.add_frame(keypoints, keypoint_scores)
                
                # 获取步态分析结果
                gait_results = gait_analyzer.get_results()
                
                # 绘制足部轨迹
                display_frame1 = gait_analyzer.draw_foot_trajectory(display_frame1)
                
                # 绘制步态指标
                display_frame1 = draw_gait_metrics(display_frame1, gait_results)
                
                # 添加FPS信息
                cv2.putText(display_frame1, f"FPS: {fps}", (10, display_frame1.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 如果不是联合显示，则单独显示
                if not args.no_display:
                    cv2.imshow('CAM0 - 步态分析', display_frame1)
                
                frames_to_show = True
            
            # 处理摄像头1的帧
            if 'frame_1' in return_dict and 'results_1' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                display_frame2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                
                # 处理姿态估计结果
                results = return_dict['results_1']
                keypoints, keypoint_scores = process_keypoints_for_gait(results, 1)
                
                # 更新步态分析器（可选，如果想对两个视角分别分析）
                # 这里我们使用摄像头0的姿态估计结果进行步态分析
                
                # 添加FPS信息
                cv2.putText(display_frame2, f"FPS: {fps}", (10, display_frame2.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 如果不是联合显示，则单独显示
                if not args.no_display:
                    cv2.imshow('CAM1 - 步态分析', display_frame2)
                
                frames_to_show = True
            
            # 创建合并视图（两个摄像头并排）
            if 'frame_0' in return_dict and 'frame_1' in return_dict:
                buffer1 = np.frombuffer(return_dict['frame_0'], dtype=np.uint8)
                buffer2 = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                
                frame1 = cv2.imdecode(buffer1, cv2.IMREAD_COLOR)
                frame2 = cv2.imdecode(buffer2, cv2.IMREAD_COLOR)
                
                # 确保两个帧的大小相同
                if frame1.shape[0] == frame2.shape[0] and frame1.shape[1] == frame2.shape[1]:
                    # 水平合并两个帧
                    combined_frame = np.hstack((frame1, frame2))
                    
                    # 获取步态分析结果
                    gait_results = gait_analyzer.get_results()
                    
                    # 绘制步态指标
                    combined_frame = draw_gait_metrics(combined_frame, gait_results)
                    
                    # 添加FPS信息
                    cv2.putText(combined_frame, f"FPS: {fps}", (10, combined_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 显示合并视图
                    if not args.no_display:
                        cv2.imshow('双摄像头步态分析', combined_frame)
                    
                    # 写入输出视频
                    if output_writer is not None:
                        output_writer.write(combined_frame)
            
            # 检查错误
            for cam_id in [0, 1]:
                if f'error_{cam_id}' in return_dict:
                    print(f"CAM {cam_id} Error: {return_dict[f'error_{cam_id}']}")
                    return_dict.pop(f'error_{cam_id}', None)
            
            # 检查退出 - 只在有帧显示时处理键盘事件
            if frames_to_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # 短暂暂停，减少CPU使用
            time.sleep(0.01)
            
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
        
        # 关闭输出视频
        if 'output_writer' in locals() and output_writer is not None:
            output_writer.release()
        
        # 打印最终结果
        if 'gait_analyzer' in locals():
            results = gait_analyzer.get_results()
            print("\n\n步态分析结果:")
            for key, value in results.items():
                print(f"{key}: {value:.2f}")
        
        # 关闭窗口
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 