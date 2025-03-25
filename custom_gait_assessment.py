import cv2
import numpy as np
import math
import argparse
import os
import time
from collections import deque
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector

def calculate_distance(p1, p2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2, p3=None):
    """计算两点或三点之间的角度
    
    如果只有两个点，计算与水平线的角度
    如果有三个点，计算三点形成的角度
    """
    if p3 is None:
        # 计算与水平线的角度
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    else:
        # 计算三点角度
        ang = math.degrees(
            math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - 
            math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        return ang + 360 if ang < 0 else ang

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
            if self.double_support_start is not None:
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

def process_video(video_path, det_model, pose_model, output_path=None, display=True):
    """处理视频，分析并可视化步态
    
    Args:
        video_path: 输入视频路径
        det_model: 人体检测模型
        pose_model: 姿态估计模型
        output_path: 输出视频路径，为None则不输出
        display: 是否显示处理过程
    
    Returns:
        gait_results: 步态分析结果
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 创建步态分析器
    gait_analyzer = GaitAnalyzer()
    
    # 跟踪的人ID (使用最先检测到的人)
    tracked_id = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\r处理帧: {frame_count}/{total_frames}", end="")
        
        # 每隔几帧处理一次 (减少计算量)
        if frame_count % 2 != 0 and frame_count > 1:
            if output_path:
                out.write(frame)
            continue
        
        # 人体检测
        det_results = inference_detector(det_model, frame)
        pred_instances = det_results.pred_instances
        
        if len(pred_instances.bboxes) == 0:
            if display:
                cv2.imshow('Gait Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if output_path:
                out.write(frame)
            continue
        
        # 找到目标人
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        
        if tracked_id is None:
            # 第一次检测，选择置信度最高的人
            tracked_id = np.argmax(scores)
        else:
            # 后续帧，选择与前一帧最近的人
            if len(bboxes) > 0:
                tracked_id = 0  # 默认第一个
        
        person_bbox = bboxes[tracked_id]
        
        # 姿态估计
        pose_results = inference_topdown(pose_model, frame, [{"bbox": person_bbox}])
        
        if not pose_results or len(pose_results) == 0:
            if display:
                cv2.imshow('Gait Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if output_path:
                out.write(frame)
            continue
        
        # 获取关键点
        keypoints = pose_results[0].pred_instances.keypoints[0].cpu().numpy()
        keypoint_scores = pose_results[0].pred_instances.keypoint_scores[0].cpu().numpy()
        
        # 添加到步态分析器
        timestamp = frame_count / fps  # 使用帧号/FPS作为时间戳
        gait_analyzer.add_frame(keypoints, keypoint_scores, timestamp)
        
        # 绘制关键点和骨架
        vis_frame = frame.copy()
        for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
            if score > 0.5:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(vis_frame, (x, y), 4, (0, 255, 0), -1)
        
        # 绘制足部轨迹
        vis_frame = gait_analyzer.draw_foot_trajectory(vis_frame)
        
        # 绘制步态指标
        gait_results = gait_analyzer.get_results()
        vis_frame = draw_gait_metrics(vis_frame, gait_results)
        
        # 显示处理后的帧
        if display:
            cv2.imshow('Gait Analysis', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 保存输出视频
        if output_path:
            out.write(vis_frame)
    
    # 释放资源
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # 返回步态分析结果
    return gait_analyzer.get_results()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='步态分析和评估工具')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--det-config', type=str, default='mmdetection/configs/rtmdet/rtmdet_m_640-8xb32_coco-person.py',
                        help='检测模型配置文件')
    parser.add_argument('--det-checkpoint', type=str, 
                        default='mmdetection/work_dirs/rtmdet_m_640-8xb32_coco-person/epoch_200.pth',
                        help='检测模型权重文件')
    parser.add_argument('--pose-config', type=str, 
                        default='configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
                        help='姿态估计模型配置文件')
    parser.add_argument('--pose-checkpoint', type=str, 
                        default='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
                        help='姿态估计模型权重文件')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备，例如cuda:0或cpu')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径')
    parser.add_argument('--no-display', action='store_true', help='不显示处理过程')
    
    args = parser.parse_args()
    
    # 初始化模型
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    
    # 处理视频
    results = process_video(args.video, det_model, pose_model, 
                           output_path=args.output, 
                           display=not args.no_display)
    
    # 打印最终结果
    print("\n\n步态分析结果:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")

if __name__ == '__main__':
    main() 