import cv2
import torch
import multiprocessing as mp
import numpy as np
import os
import argparse
import sys
import time
from collections import deque
import math

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

def calculate_distance(p1, p2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def compute_custom_keypoints(keypoints, keypoint_scores, thr=0.3):
    """计算表格评估所需的自定义关键点
    
    Args:
        keypoints: 关键点坐标数组，形状为(N, 2)
        keypoint_scores: 关键点置信度分数，形状为(N,)
        thr: 关键点有效性阈值
    
    Returns:
        dict: 包含所有自定义关键点的字典
    """
    # 关键点索引 (根据RTMPose全身模型)
    NOSE_IDX = 0
    LEFT_EYE_IDX = 1
    RIGHT_EYE_IDX = 2
    LEFT_EAR_IDX = 3
    RIGHT_EAR_IDX = 4
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6
    LEFT_ELBOW_IDX = 7
    RIGHT_ELBOW_IDX = 8
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    LEFT_HIP_IDX = 11
    RIGHT_HIP_IDX = 12
    LEFT_KNEE_IDX = 13
    RIGHT_KNEE_IDX = 14
    LEFT_ANKLE_IDX = 15
    RIGHT_ANKLE_IDX = 16
    
    # 验证关键点有效性
    def is_valid(idx):
        return idx < len(keypoint_scores) and keypoint_scores[idx] > thr
    
    # 获取有效关键点
    valid_keypoints = {}
    for idx in range(len(keypoint_scores)):
        if is_valid(idx):
            valid_keypoints[idx] = keypoints[idx]
    
    # 自定义关键点字典
    custom_points = {}
    
    # 初始化所有需要的关键点
    # N1-N17对应表格中需要的点
    
    # N1: 乳突(左右耳和肩膀中点计算)
    if is_valid(LEFT_EAR_IDX) and is_valid(LEFT_SHOULDER_IDX):
        x = valid_keypoints[LEFT_EAR_IDX][0] * 0.8 + valid_keypoints[LEFT_SHOULDER_IDX][0] * 0.2
        y = valid_keypoints[LEFT_EAR_IDX][1] * 0.9 + valid_keypoints[LEFT_SHOULDER_IDX][1] * 0.1
        custom_points['N1'] = (int(x), int(y))
    elif is_valid(RIGHT_EAR_IDX) and is_valid(RIGHT_SHOULDER_IDX):
        x = valid_keypoints[RIGHT_EAR_IDX][0] * 0.8 + valid_keypoints[RIGHT_SHOULDER_IDX][0] * 0.2
        y = valid_keypoints[RIGHT_EAR_IDX][1] * 0.9 + valid_keypoints[RIGHT_SHOULDER_IDX][1] * 0.1
        custom_points['N1'] = (int(x), int(y))
    
    # N2: 肩峰(侧面)
    if is_valid(LEFT_SHOULDER_IDX):
        custom_points['N2_left'] = (int(valid_keypoints[LEFT_SHOULDER_IDX][0]), 
                                    int(valid_keypoints[LEFT_SHOULDER_IDX][1]))
    if is_valid(RIGHT_SHOULDER_IDX):
        custom_points['N2_right'] = (int(valid_keypoints[RIGHT_SHOULDER_IDX][0]), 
                                     int(valid_keypoints[RIGHT_SHOULDER_IDX][1]))
    
    # 计算肩膀中点(用于后续计算)
    shoulder_mid = None
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(RIGHT_SHOULDER_IDX):
        mid_x = (valid_keypoints[LEFT_SHOULDER_IDX][0] + valid_keypoints[RIGHT_SHOULDER_IDX][0]) / 2
        mid_y = (valid_keypoints[LEFT_SHOULDER_IDX][1] + valid_keypoints[RIGHT_SHOULDER_IDX][1]) / 2
        shoulder_mid = (int(mid_x), int(mid_y))
    
    # 计算髋部中点(用于后续计算)
    hip_mid = None
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX):
        mid_x = (valid_keypoints[LEFT_HIP_IDX][0] + valid_keypoints[RIGHT_HIP_IDX][0]) / 2
        mid_y = (valid_keypoints[LEFT_HIP_IDX][1] + valid_keypoints[RIGHT_HIP_IDX][1]) / 2
        hip_mid = (int(mid_x), int(mid_y))
        custom_points['N8'] = hip_mid  # N8: 髂前上棘
    
    # N3: 第一胸椎(颈部下方)
    if shoulder_mid and is_valid(NOSE_IDX):
        # 在肩膀中点和鼻子之间取一点作为颈椎中点
        neck_x = shoulder_mid[0] * 0.7 + valid_keypoints[NOSE_IDX][0] * 0.3
        neck_y = shoulder_mid[1] * 0.8 + valid_keypoints[NOSE_IDX][1] * 0.2
        custom_points['neck_mid'] = (int(neck_x), int(neck_y))
        
        # 第一胸椎在颈椎中点下方
        x = neck_x
        y = neck_y + (shoulder_mid[1] - neck_y) * 0.3
        custom_points['N3'] = (int(x), int(y))
    
    # 如果有肩膀中点和髋部中点，计算脊柱上的点
    if shoulder_mid and hip_mid:
        # 计算脊柱线(从肩膀中点到髋部中点)
        spine_length = calculate_distance(shoulder_mid, hip_mid)
        spine_angle = calculate_angle(shoulder_mid, hip_mid)
        
        # N4: 第十二胸椎(脊柱上点)
        ratio = 0.6  # 大约位于肩膀和髋部之间60%处
        x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
        y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
        custom_points['N4'] = (int(x), int(y))
        
        # N5: 第一腰椎(脊柱上点)
        ratio = 0.7  # 大约位于肩膀和髋部之间70%处
        x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
        y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
        custom_points['N5'] = (int(x), int(y))
        
        # N6: 第五腰椎(髋部中点上方)
        ratio = 0.9  # 大约位于肩膀和髋部之间90%处
        x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
        y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
        custom_points['N6'] = (int(x), int(y))
        
        # N13: 脊柱棘突(沿脊柱线的多个点)
        spine_points = []
        for i in range(5):
            ratio = 0.2 + i * 0.15  # 均匀分布在脊柱上
            x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
            y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
            spine_points.append((int(x), int(y)))
        custom_points['N13'] = spine_points
    
    # N7: 耻骨联合(髋部中点下方)
    if hip_mid:
        # 耻骨联合在髋部下方一定距离
        x = hip_mid[0]
        y = hip_mid[1] + (hip_mid[1] - shoulder_mid[1]) * 0.15 if shoulder_mid else hip_mid[1] + 20
        custom_points['N7'] = (int(x), int(y))
    
    # N9: 髌骨(膝关节)
    if is_valid(LEFT_KNEE_IDX):
        custom_points['N9_left'] = (int(valid_keypoints[LEFT_KNEE_IDX][0]), 
                                   int(valid_keypoints[LEFT_KNEE_IDX][1]))
    if is_valid(RIGHT_KNEE_IDX):
        custom_points['N9_right'] = (int(valid_keypoints[RIGHT_KNEE_IDX][0]), 
                                    int(valid_keypoints[RIGHT_KNEE_IDX][1]))
    
    # N10: 第五跖骨(足踝外侧)
    if is_valid(LEFT_ANKLE_IDX):
        angle = 45 if is_valid(LEFT_KNEE_IDX) else 0
        dist = 15  # 距离足踝的像素距离
        x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        custom_points['N10_left'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        angle = -45 if is_valid(RIGHT_KNEE_IDX) else 0
        dist = 15
        x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        custom_points['N10_right'] = (int(x), int(y))
    
    # N11: 足跟(足踝后方)
    if is_valid(LEFT_ANKLE_IDX):
        if is_valid(LEFT_KNEE_IDX):
            angle = calculate_angle(valid_keypoints[LEFT_KNEE_IDX], valid_keypoints[LEFT_ANKLE_IDX])
            dist = 20  # 距离足踝的像素距离
            x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle+180))
            y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle+180))
        else:
            x = valid_keypoints[LEFT_ANKLE_IDX][0] - 20
            y = valid_keypoints[LEFT_ANKLE_IDX][1]
        custom_points['N11_left'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        if is_valid(RIGHT_KNEE_IDX):
            angle = calculate_angle(valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
            dist = 20
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle+180))
            y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle+180))
        else:
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] - 20
            y = valid_keypoints[RIGHT_ANKLE_IDX][1]
        custom_points['N11_right'] = (int(x), int(y))
    
    # N12: 手掌(手腕前方)
    if is_valid(LEFT_WRIST_IDX):
        if is_valid(LEFT_ELBOW_IDX):
            angle = calculate_angle(valid_keypoints[LEFT_ELBOW_IDX], valid_keypoints[LEFT_WRIST_IDX])
            dist = 25
            x = valid_keypoints[LEFT_WRIST_IDX][0] + dist * math.cos(math.radians(angle))
            y = valid_keypoints[LEFT_WRIST_IDX][1] + dist * math.sin(math.radians(angle))
        else:
            x = valid_keypoints[LEFT_WRIST_IDX][0] + 25
            y = valid_keypoints[LEFT_WRIST_IDX][1]
        custom_points['N12_left'] = (int(x), int(y))
    
    if is_valid(RIGHT_WRIST_IDX):
        if is_valid(RIGHT_ELBOW_IDX):
            angle = calculate_angle(valid_keypoints[RIGHT_ELBOW_IDX], valid_keypoints[RIGHT_WRIST_IDX])
            dist = 25
            x = valid_keypoints[RIGHT_WRIST_IDX][0] + dist * math.cos(math.radians(angle))
            y = valid_keypoints[RIGHT_WRIST_IDX][1] + dist * math.sin(math.radians(angle))
        else:
            x = valid_keypoints[RIGHT_WRIST_IDX][0] + 25
            y = valid_keypoints[RIGHT_WRIST_IDX][1]
        custom_points['N12_right'] = (int(x), int(y))
    
    # N14: 双侧髂嵴(髋部两侧)
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX):
        hip_center_x = (valid_keypoints[LEFT_HIP_IDX][0] + valid_keypoints[RIGHT_HIP_IDX][0]) / 2
        hip_center_y = (valid_keypoints[LEFT_HIP_IDX][1] + valid_keypoints[RIGHT_HIP_IDX][1]) / 2
        hip_width = calculate_distance(valid_keypoints[LEFT_HIP_IDX], valid_keypoints[RIGHT_HIP_IDX])
        
        left_x = valid_keypoints[LEFT_HIP_IDX][0] - hip_width * 0.15
        left_y = valid_keypoints[LEFT_HIP_IDX][1]
        right_x = valid_keypoints[RIGHT_HIP_IDX][0] + hip_width * 0.15
        right_y = valid_keypoints[RIGHT_HIP_IDX][1]
        
        custom_points['N14_left'] = (int(left_x), int(left_y))
        custom_points['N14_right'] = (int(right_x), int(right_y))
    
    # N15: 髂后上嵴(髋部后方)
    if hip_mid:
        # 髂后上嵴位于髋部中点后方
        back_dist = 30  # 向后的距离
        x = hip_mid[0] - back_dist
        y = hip_mid[1]
        custom_points['N15'] = (int(x), int(y))
    
    # N16: 第一跖骨(足踝内侧)
    if is_valid(LEFT_ANKLE_IDX):
        angle = -45 if is_valid(LEFT_KNEE_IDX) else 0
        dist = 15
        x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        custom_points['N16_left'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        angle = 45 if is_valid(RIGHT_KNEE_IDX) else 0
        dist = 15
        x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        custom_points['N16_right'] = (int(x), int(y))
    
    # N17: 足尖(足踝前方)
    if is_valid(LEFT_ANKLE_IDX):
        if is_valid(LEFT_KNEE_IDX):
            angle = calculate_angle(valid_keypoints[LEFT_KNEE_IDX], valid_keypoints[LEFT_ANKLE_IDX])
            dist = 40
            x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
            y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        else:
            x = valid_keypoints[LEFT_ANKLE_IDX][0] + 40
            y = valid_keypoints[LEFT_ANKLE_IDX][1]
        custom_points['N17_left'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        if is_valid(RIGHT_KNEE_IDX):
            angle = calculate_angle(valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
            dist = 40
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
            y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        else:
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] + 40
            y = valid_keypoints[RIGHT_ANKLE_IDX][1]
        custom_points['N17_right'] = (int(x), int(y))
    
    return custom_points

def draw_custom_keypoints(image, custom_points, color_map=None, radius=8, thickness=3):
    """绘制自定义关键点
    
    Args:
        image: 输入图像
        custom_points: 包含自定义关键点的字典
        color_map: 关键点颜色映射，如果为None则使用默认颜色
        radius: 关键点圆的半径
        thickness: 线条粗细
    
    Returns:
        image: 绘制了关键点的图像
    """
    if color_map is None:
        # 默认颜色映射 - 使用更明亮的颜色
        color_map = {
            'N1': (0, 0, 255),      # 乳突 - 红色
            'N2_left': (0, 255, 0), # 肩峰 - 绿色
            'N2_right': (0, 255, 0),
            'N3': (255, 0, 0),      # 第一胸椎 - 蓝色
            'N4': (0, 255, 255),    # 第十二胸椎 - 黄色
            'N5': (255, 0, 255),    # 第一腰椎 - 洋红色
            'N6': (255, 255, 0),    # 第五腰椎 - 青色
            'N7': (128, 0, 128),    # 耻骨联合 - 紫色
            'N8': (255, 165, 0),    # 髂前上棘 - 橙色
            'N9_left': (255, 192, 203), # 髌骨 - 粉色
            'N9_right': (255, 192, 203),
            'N10_left': (0, 128, 0),    # 第五跖骨 - 深绿色
            'N10_right': (0, 128, 0),
            'N11_left': (139, 69, 19),  # 足跟 - 棕色
            'N11_right': (139, 69, 19),
            'N12_left': (70, 130, 180), # 手掌 - 钢蓝色
            'N12_right': (70, 130, 180),
            'N13': (255, 255, 255),     # 脊柱棘突 - 白色
            'N14_left': (230, 230, 250), # 双侧髂嵴 - 淡紫色
            'N14_right': (230, 230, 250),
            'N15': (255, 250, 205),     # 髂后上嵴 - 米色
            'N16_left': (173, 216, 230), # 第一跖骨 - 淡蓝色
            'N16_right': (173, 216, 230),
            'N17_left': (152, 251, 152), # 足尖 - 淡绿色
            'N17_right': (152, 251, 152),
            'neck_mid': (255, 99, 71)   # 颈椎中点 - 番茄色
        }
    
    # 计数绘制的关键点数
    point_count = 0
    
    # 绘制所有关键点
    for key, point in custom_points.items():
        color = color_map.get(key, (0, 0, 255))
        
        if key == 'N13' and isinstance(point, list):
            # 脊柱棘突是多个点，连接它们
            for i in range(len(point) - 1):
                cv2.line(image, point[i], point[i+1], color, thickness)
                cv2.circle(image, point[i], radius, color, -1)
                point_count += 1
            if point:
                cv2.circle(image, point[-1], radius, color, -1)
                point_count += 1
        else:
            cv2.circle(image, point, radius, color, -1)
            point_count += 1
            
            # 标注关键点
            if not key.endswith('_left') and not key.endswith('_right'):
                cv2.putText(image, key, (point[0] + 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # 对于左右侧的关键点，只显示基本名称
                base_key = key.split('_')[0]
                cv2.putText(image, base_key, (point[0] + 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 绘制连接线以显示脊柱曲线
    spine_keys = ['N3', 'N4', 'N5', 'N6', 'N8']
    spine_points = []
    
    for key in spine_keys:
        if key in custom_points:
            spine_points.append(custom_points[key])
    
    # 连接脊柱关键点
    for i in range(len(spine_points) - 1):
        cv2.line(image, spine_points[i], spine_points[i+1], (255, 255, 255), thickness+1)  # 使用更粗的白色线
    
    # 添加有关绘制的关键点数量的信息
    cv2.putText(image, f"关键点数: {point_count}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image

def process_keypoints_for_custom_visualization(results, camera_idx):
    """处理关键点数据用于可视化和步态分析
    
    Args:
        results: RTMPose的关键点检测结果
        camera_idx: 相机索引
    
    Returns:
        keypoints, keypoint_scores, person_idx: 处理后的关键点坐标、分数和人物索引
    """
    # 打印调试信息
    if config.SystemConfig.DEBUG_MODE:
        print(f"CAM {camera_idx} - 处理关键点数据，结果类型: {type(results)}")
        if results:
            if isinstance(results, list):
                print(f"CAM {camera_idx} - 结果列表长度: {len(results)}")
                if results and isinstance(results[0], dict):
                    print(f"CAM {camera_idx} - 第一个结果的键: {list(results[0].keys())}")
    
    # 先检查结果是否为空
    if not results:
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 结果为空")
        return None, None, -1
    
    # 检查结果类型并适当处理
    if isinstance(results, list):
        # 如果results是一个列表
        if not results:
            return None, None, -1
        
        # 深入检查结果结构
        first_item = results[0]
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 第一个结果类型: {type(first_item)}")
        
        # MMPose可能返回不同的结构，需要适配
        if isinstance(first_item, dict):
            # 可能是以下结构之一:
            # 1. [{'predictions': [...]}]
            # 2. [{'instances': [...]}, {...}]
            if 'predictions' in first_item:
                predictions = first_item.get('predictions', [])
                if config.SystemConfig.DEBUG_MODE:
                    print(f"CAM {camera_idx} - 找到predictions键，数据类型: {type(predictions)}")
            elif 'instances' in first_item:
                predictions = first_item.get('instances', [])
                if config.SystemConfig.DEBUG_MODE:
                    print(f"CAM {camera_idx} - 找到instances键，数据类型: {type(predictions)}")
            else:
                # 尝试其他可能的键
                possible_keys = ['pred_instances', 'keypoints', 'poses']
                for key in possible_keys:
                    if key in first_item:
                        predictions = first_item.get(key, [])
                        if config.SystemConfig.DEBUG_MODE:
                            print(f"CAM {camera_idx} - 找到{key}键，数据类型: {type(predictions)}")
                        break
                else:
                    # 如果找不到任何已知键，就尝试使用第一个键的值
                    if first_item:
                        first_key = list(first_item.keys())[0]
                        predictions = first_item.get(first_key, [])
                        if config.SystemConfig.DEBUG_MODE:
                            print(f"CAM {camera_idx} - 使用第一个键{first_key}，数据类型: {type(predictions)}")
                    else:
                        predictions = []
        elif isinstance(first_item, list):
            # 可能是嵌套列表 [[{...}, {...}]]
            predictions = first_item
            if config.SystemConfig.DEBUG_MODE:
                print(f"CAM {camera_idx} - 列表中有嵌套列表，长度: {len(predictions)}")
        else:
            # 可能是其他类型，直接使用整个结果列表
            predictions = results
            if config.SystemConfig.DEBUG_MODE:
                print(f"CAM {camera_idx} - 直接使用结果列表，长度: {len(predictions)}")
    else:
        # 如果不是列表，可能直接是单个结果
        predictions = [results]
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 单个结果对象，类型: {type(results)}")
    
    # 再次检查predictions的类型，如果还是不正确，尝试强制转为列表
    if not isinstance(predictions, list) and hasattr(predictions, '__iter__'):
        predictions = list(predictions)
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 将可迭代对象转为列表，长度: {len(predictions)}")
    
    # 确保predictions非空
    if not predictions:
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 处理后predictions为空")
        return None, None, -1
    
    # 打印第一个prediction的内容以便调试
    if config.SystemConfig.DEBUG_MODE and predictions:
        first_pred = predictions[0]
        print(f"CAM {camera_idx} - 第一个prediction类型: {type(first_pred)}")
        if isinstance(first_pred, dict):
            print(f"CAM {camera_idx} - 第一个prediction键: {list(first_pred.keys())}")
    
    # 寻找最佳的人物
    best_person_idx = 0
    max_score = -1
    found_valid = False
    
    for i, pred in enumerate(predictions):
        # 尝试不同的获取方式
        keypoints = None
        keypoint_scores = None
        
        if isinstance(pred, dict):
            # 字典类型，直接获取
            # 尝试不同的键名 (MMPose不同版本可能使用不同的键)
            for kpts_key in ['keypoints', 'keypoint', 'kpts', 'pose', 'poses']:
                if kpts_key in pred:
                    keypoints = pred.get(kpts_key)
                    break
            
            for score_key in ['keypoint_scores', 'scores', 'score', 'kpt_scores']:
                if score_key in pred:
                    keypoint_scores = pred.get(score_key)
                    break
            
            # 如果没有找到，但找到了一个numpy数组类型的值，可能是关键点
            if keypoints is None:
                for k, v in pred.items():
                    if isinstance(v, np.ndarray) and len(v.shape) == 2:
                        keypoints = v
                        if config.SystemConfig.DEBUG_MODE:
                            print(f"CAM {camera_idx} - 使用{k}作为关键点，形状: {v.shape}")
                        break
        else:
            # 其他类型，尝试获取属性
            try:
                # 常见的属性名
                for attr_name in ['keypoints', 'keypoint', 'kpts', 'pose', 'poses']:
                    if hasattr(pred, attr_name):
                        keypoints = getattr(pred, attr_name)
                        break
                
                # 分数属性
                for attr_name in ['keypoint_scores', 'scores', 'score', 'kpt_scores']:
                    if hasattr(pred, attr_name):
                        keypoint_scores = getattr(pred, attr_name)
                        break
            except:
                continue
        
        # 如果找到了关键点但没有分数，尝试生成统一分数
        if keypoints is not None and keypoint_scores is None:
            # 默认给所有关键点一个中等置信度分数
            if isinstance(keypoints, np.ndarray) and len(keypoints.shape) == 2:
                keypoint_scores = np.ones(len(keypoints)) * 0.7
                if config.SystemConfig.DEBUG_MODE:
                    print(f"CAM {camera_idx} - 生成默认关键点分数")
        
        # 打印调试信息
        if config.SystemConfig.DEBUG_MODE and keypoints is not None:
            print(f"CAM {camera_idx} - 第{i}个人物，关键点形状: {np.shape(keypoints) if isinstance(keypoints, np.ndarray) else 'not numpy'}")
            if keypoint_scores is not None:
                print(f"CAM {camera_idx} - 第{i}个人物，分数形状: {np.shape(keypoint_scores) if isinstance(keypoint_scores, np.ndarray) else 'not numpy'}")
        
        if keypoints is not None and keypoint_scores is not None:
            try:
                # 计算平均分数
                avg_score = np.mean(keypoint_scores)
                if avg_score > max_score:
                    max_score = avg_score
                    best_person_idx = i
                    found_valid = True
                    if config.SystemConfig.DEBUG_MODE:
                        print(f"CAM {camera_idx} - 找到有效人物，平均分数: {avg_score:.2f}")
            except Exception as e:
                if config.SystemConfig.DEBUG_MODE:
                    print(f"CAM {camera_idx} - 计算分数时出错: {str(e)}")
                continue
    
    if not found_valid:
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 未找到有效人物")
        return None, None, -1
    
    # 获取最佳人物的关键点
    best_pred = predictions[best_person_idx]
    
    # 根据对象类型获取关键点和分数
    keypoints = None
    keypoint_scores = None
    
    if isinstance(best_pred, dict):
        # 与前面相同，尝试不同的键名
        for kpts_key in ['keypoints', 'keypoint', 'kpts', 'pose', 'poses']:
            if kpts_key in best_pred:
                keypoints = best_pred.get(kpts_key)
                break
        
        for score_key in ['keypoint_scores', 'scores', 'score', 'kpt_scores']:
            if score_key in best_pred:
                keypoint_scores = best_pred.get(score_key)
                break
                
        # 如果没有找到，但找到了一个numpy数组类型的值，可能是关键点
        if keypoints is None:
            for k, v in best_pred.items():
                if isinstance(v, np.ndarray) and len(v.shape) == 2:
                    keypoints = v
                    break
    else:
        try:
            # 与前面相同，尝试不同的属性名
            for attr_name in ['keypoints', 'keypoint', 'kpts', 'pose', 'poses']:
                if hasattr(best_pred, attr_name):
                    keypoints = getattr(best_pred, attr_name)
                    break
            
            for attr_name in ['keypoint_scores', 'scores', 'score', 'kpt_scores']:
                if hasattr(best_pred, attr_name):
                    keypoint_scores = getattr(best_pred, attr_name)
                    break
        except:
            return None, None, -1
    
    # 如果找到了关键点但没有分数，尝试生成统一分数
    if keypoints is not None and keypoint_scores is None:
        # 默认给所有关键点一个中等置信度分数
        if isinstance(keypoints, np.ndarray) and len(keypoints.shape) == 2:
            keypoint_scores = np.ones(len(keypoints)) * 0.7
    
    if keypoints is None or keypoint_scores is None:
        if config.SystemConfig.DEBUG_MODE:
            print(f"CAM {camera_idx} - 无法提取最佳人物的关键点或分数")
        return None, None, -1
    
    # 打印最终提取的关键点信息
    if config.SystemConfig.DEBUG_MODE:
        if isinstance(keypoints, np.ndarray):
            print(f"CAM {camera_idx} - 最终关键点: 形状{keypoints.shape}, 类型{type(keypoints)}")
        if isinstance(keypoint_scores, np.ndarray):
            print(f"CAM {camera_idx} - 最终分数: 形状{keypoint_scores.shape}, 类型{type(keypoint_scores)}")
    
    return keypoints, keypoint_scores, best_person_idx

def custom_process_frame(frame, pose_results, camera_idx):
    """处理帧，应用自定义关键点可视化
    
    Args:
        frame: 输入图像帧
        pose_results: 姿态估计结果
        camera_idx: 摄像头索引
    
    Returns:
        display_frame: 处理后的显示帧
    """
    display_frame = frame.copy()
    
    try:
        # 处理关键点
        keypoints, keypoint_scores, person_idx = process_keypoints_for_custom_visualization(pose_results, camera_idx)
        
        if keypoints is not None and keypoint_scores is not None:
            # 打印一些调试信息
            if config.SystemConfig.DEBUG_MODE:
                print(f"CAM {camera_idx} - 检测到关键点: {len(keypoints)}, 平均置信度: {np.mean(keypoint_scores):.2f}")
            
            # 计算自定义关键点
            custom_points = compute_custom_keypoints(keypoints, keypoint_scores)
            
            # 打印一些调试信息
            if config.SystemConfig.DEBUG_MODE:
                print(f"CAM {camera_idx} - 生成自定义关键点: {len(custom_points)}")
            
            # 绘制原始关键点
            for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                if score > 0.3:  # 使用较低的阈值显示更多关键点
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
                    # 可选：显示关键点索引用于调试
                    # cv2.putText(display_frame, str(i), (x + 3, y - 3), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制自定义关键点
            display_frame = draw_custom_keypoints(display_frame, custom_points)
            
            # 添加相机标识和时间戳
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"CAM {camera_idx} | {timestamp}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 没有检测到有效的人物关键点
            cv2.putText(display_frame, f"CAM {camera_idx} | 未检测到人体", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
    except Exception as e:
        # 出现异常时输出错误信息
        print(f"自定义关键点处理错误: {str(e)}")
        cv2.putText(display_frame, f"CAM {camera_idx} | 处理错误", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 调试信息
        if config.SystemConfig.DEBUG_MODE:
            import traceback
            traceback.print_exc()
    
    return display_frame

def camera_process_with_custom_viz(camera_id, return_dict, shared_data, model_name='rtmpose-l_8xb32-270e_coco-wholebody-384x288', device='cuda:0'):
    """摄像头处理进程，包含自定义关键点可视化
    
    Args:
        camera_id: 摄像头ID
        return_dict: 用于进程间通信的共享字典
        shared_data: 共享数据
        model_name: 使用的模型名称
        device: 设备名称
    """
    try:
        # 导入多进程摄像头模块
        from .multiprocess_camera import camera_process
        # 调用原始camera_process函数但增加额外的处理
        camera_process(camera_id, return_dict, shared_data, model_name, device, 
                       custom_frame_processor=custom_process_frame)
    except Exception as e:
        print(f"摄像头进程出错: {str(e)}")
        return_dict[f'error_{camera_id}'] = str(e)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双摄像头姿态估计系统，带自定义关键点可视化')
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
        
        # 获取模型名称
        model_name = args.model if args.model else config.ModelConfig.DEFAULT_MODEL
        
        # 创建两个摄像头处理进程
        process1 = mp.Process(target=camera_process_with_custom_viz, 
                             args=(0, return_dict, shared_data, model_name, device))
        process2 = mp.Process(target=camera_process_with_custom_viz, 
                             args=(1, return_dict, shared_data, model_name, device))
        
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
                cv2.imshow('摄像头 0 - 自定义关键点', display_frame1)
                frames_to_show = True
                
            if 'frame_1' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                display_frame2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('摄像头 1 - 自定义关键点', display_frame2)
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