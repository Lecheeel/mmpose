#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
import math

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


# 定义配置常量
# 模型配置文件和权重的URL
MODEL_CONFIGS = {
    # 人体检测相关配置
    'det': {
        'config': 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py',
        'checkpoint': 'C:/Users/ryanc/mmpose-main/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        'cat_id': 0,           # 人体检测的类别ID
        'bbox_thr': 0.3,       # 边界框检测阈值
        'nms_thr': 0.4,        # 非极大值抑制阈值
    },
    
    # 姿态估计相关配置
    'pose': {
        'config': 'configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py',
        'checkpoint': 'C:/Users/ryanc/mmpose-main/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
        'kpt_thr': 0.5,         # 关键点检测阈值
        'skeleton_style': 'mmpose',
        'radius': 5,
        'thickness': 3,
        'alpha': 0.8,
        'show_posture_analysis': True  # 是否显示姿态分析结果
    }
}


# 定义所有配置参数类
class Config:
    """全局配置类，存储所有参数"""
    
    def __init__(self):
        # 设备配置
        self.device = 'cuda:0'  # 使用GPU (cuda:0) 或 CPU ('cpu')
        
        # 人体检测器配置
        self.det_config = MODEL_CONFIGS['det']['config']
        self.det_checkpoint = MODEL_CONFIGS['det']['checkpoint']
        self.det_cat_id = MODEL_CONFIGS['det']['cat_id']
        self.bbox_thr = MODEL_CONFIGS['det']['bbox_thr']
        self.nms_thr = MODEL_CONFIGS['det']['nms_thr']
        
        # 姿态估计配置
        self.pose_config = MODEL_CONFIGS['pose']['config']
        self.pose_checkpoint = MODEL_CONFIGS['pose']['checkpoint']
        self.kpt_thr = MODEL_CONFIGS['pose']['kpt_thr']
        self.skeleton_style = MODEL_CONFIGS['pose']['skeleton_style']
        self.radius = MODEL_CONFIGS['pose']['radius']
        self.thickness = MODEL_CONFIGS['pose']['thickness']
        self.alpha = MODEL_CONFIGS['pose']['alpha']
        self.show_posture_analysis = MODEL_CONFIGS['pose']['show_posture_analysis']
        
        # 可视化配置
        self.draw_heatmap = False     # 是否绘制热图
        self.show_kpt_idx = False     # 是否显示关键点索引
        self.draw_bbox = False        # 是否绘制边界框
        self.radius = 5               # 关键点半径
        self.thickness = 3            # 线条粗细
        self.alpha = 0.8              # 透明度
        
        # 输出配置
        self.show = True              # 是否显示结果
        self.output_root = ''         # 输出根目录，为空则不保存输出
        self.save_predictions = False # 是否保存预测结果
        
        # 性能指标
        self.fps = False               # 是否显示FPS
        
        # 可视化过滤选项
        self.draw_hands = False       # 不绘制手部关键点
        self.draw_face = False        # 不绘制面部关键点，只保留鼻子和耳朵
        
        # 新增关键点显示选项
        self.draw_iliac_midpoint = True      # 显示髂前上棘连线中点
        self.draw_neck_midpoint = True       # 显示颈椎中点（喉结处）
        self.custom_keypoint_radius = 6      # 自定义关键点半径（比普通关键点稍大）
        self.custom_keypoint_thickness = 4   # 自定义连接线粗细（比普通线条稍粗）


def filter_keypoints(data_samples, args=None):
    """过滤关键点，只保留需要的关键点（鼻子、双耳、身体和足部，去除手部和其他面部关键点）"""
    if hasattr(data_samples, 'pred_instances'):
        for i in range(len(data_samples.pred_instances)):
            # 获取关键点和分数
            keypoints = data_samples.pred_instances.keypoints[i]
            keypoint_scores = data_samples.pred_instances.keypoint_scores[i]
            
            # 需要保留的关键点索引列表（0-16: 身体关键点, 17-22: 足部关键点）
            # 面部只保留鼻子和耳朵 (索引0, 3, 4)
            remove_indices = [1, 2]  # 移除眼睛关键点 (索引1, 2为左右眼)
            
            # 移除除鼻子和耳朵外的面部关键点 (索引23-90)
            for idx in range(23, 91):
                remove_indices.append(idx)
            
            # 移除手部关键点 (索引91-132)
            for idx in range(91, 133):
                remove_indices.append(idx)
            
            # 设置需要移除的关键点的置信度为0，坐标为无效值
            for idx in remove_indices:
                if idx < len(keypoint_scores):
                    keypoint_scores[idx] = 0.0
                    keypoints[idx, :] = -1
            
            # 更新数据
            data_samples.pred_instances.keypoints[i] = keypoints
            data_samples.pred_instances.keypoint_scores[i] = keypoint_scores
            
            # 新增关键点的列表
            new_keypoints = []
            new_keypoint_scores = []
            
            # 添加两个新的估算关键点
            # 假设骨盆关键点（髋部）在身体关键点中的索引为11和12
            # 注意：实际索引取决于具体模型，可能需要调整
            HIP_LEFT_IDX = 11
            HIP_RIGHT_IDX = 12
            
            # 假设颈部关键点在身体关键点中的索引为5和6
            # 注意：可能需要根据实际模型调整
            NECK_BASE_IDX = 5  # 肩膀中心或胸部顶端
            NOSE_IDX = 0  # 鼻子
            LEFT_SHOULDER_IDX = 5  # 左肩
            RIGHT_SHOULDER_IDX = 6  # 右肩
            
            # 1. 计算髂前上棘连线中点
            # 根据配置选项决定是否添加髂前上棘中点
            if args is None or args.draw_iliac_midpoint:
                # 检查左右髋部关键点是否有效
                if (keypoint_scores[HIP_LEFT_IDX] > 0.5 and 
                    keypoint_scores[HIP_RIGHT_IDX] > 0.5):
                    # 计算髋部中点
                    iliac_x = (keypoints[HIP_LEFT_IDX, 0] + keypoints[HIP_RIGHT_IDX, 0]) / 2
                    iliac_y = (keypoints[HIP_LEFT_IDX, 1] + keypoints[HIP_RIGHT_IDX, 1]) / 2
                    
                    # 添加到新关键点列表
                    new_keypoints.append([iliac_x, iliac_y])
                    # 添加对应的置信度分数
                    avg_score = (keypoint_scores[HIP_LEFT_IDX] + keypoint_scores[HIP_RIGHT_IDX]) / 2
                    new_keypoint_scores.append(avg_score)
            
            # 2. 计算颈椎中点（喉结处）
            if args is None or args.draw_neck_midpoint:
                # 检查是否能获取有效的肩膀和鼻子关键点
                valid_shoulders = (keypoint_scores[LEFT_SHOULDER_IDX] > 0.5 and 
                                  keypoint_scores[RIGHT_SHOULDER_IDX] > 0.5)
                valid_nose = keypoint_scores[NOSE_IDX] > 0.5
                
                if valid_shoulders and valid_nose:
                    # 首先计算左右肩膀的水平中点
                    shoulder_mid_x = (keypoints[LEFT_SHOULDER_IDX, 0] + keypoints[RIGHT_SHOULDER_IDX, 0]) / 2
                    shoulder_mid_y = (keypoints[LEFT_SHOULDER_IDX, 1] + keypoints[RIGHT_SHOULDER_IDX, 1]) / 2
                    
                    # 然后计算喉结位置 - 在肩膀中点和鼻子之间的位置
                    # 通常喉结在肩膀中点到鼻子连线的下三分之一处
                    neck_midpoint_x = shoulder_mid_x + (keypoints[NOSE_IDX, 0] - shoulder_mid_x) / 3
                    neck_midpoint_y = shoulder_mid_y + (keypoints[NOSE_IDX, 1] - shoulder_mid_y) / 3
                    
                    # 添加到新关键点列表
                    new_keypoints.append([neck_midpoint_x, neck_midpoint_y])
                    # 添加对应的置信度分数
                    avg_score = (keypoint_scores[LEFT_SHOULDER_IDX] + 
                                keypoint_scores[RIGHT_SHOULDER_IDX] + 
                                keypoint_scores[NOSE_IDX]) / 3
                    new_keypoint_scores.append(avg_score)
            
            # 记录新添加的关键点数量，用于可视化时识别
            # 注意：这里我们不修改原始数组，而是在数据样本中存储新关键点
            if new_keypoints:
                # 将新关键点和分数转换为numpy数组
                new_keypoints_array = np.array(new_keypoints)
                new_keypoint_scores_array = np.array(new_keypoint_scores)
                
                # 将新关键点和分数存储到数据样本中，供后续处理
                if not hasattr(data_samples, 'custom_keypoints'):
                    data_samples.custom_keypoints = []
                    data_samples.custom_keypoint_scores = []
                
                data_samples.custom_keypoints.append(new_keypoints_array)
                data_samples.custom_keypoint_scores.append(new_keypoint_scores_array)
    
    return data_samples


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

def analyze_body_posture(keypoints, keypoint_scores, custom_keypoints=None, return_keypoints=False):
    """分析身体姿态并计算各种体态测量值
    
    Args:
        keypoints: 关键点坐标数组，形状为(N, 2)
        keypoint_scores: 关键点置信度，形状为(N,)
        custom_keypoints: 自定义关键点列表，可选
        return_keypoints: 如果为True，返回自定义关键点而不是分析结果
    
    Returns:
        dict: 包含各种体态测量值的字典，或者自定义关键点列表（如果return_keypoints=True）
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
    
    # 计算自定义关键点索引
    ILIAC_MIDPOINT_IDX = 0  # 髂前上棘连线中点
    NECK_MIDPOINT_IDX = 1   # 颈椎中点（喉结处）
    
    results = {
        '头前倾角': None,
        '头侧倾角': None,
        '头旋转角': None,
        '肩倾斜角': None,
        '圆肩角': None,
        '背部角': None,
        '腹部肥胖度': None,
        '腰曲度': None,
        '骨盆前倾角': None,
        '侧中位度': None,
        '腿型-左腿': None,
        '腿型-右腿': None,
        '左膝评估角': None,
        '右膝评估角': None,
        '身体倾斜度': None,
        '足八角': None
    }
    
    # 只有当关键点有效时才计算
    valid_keypoints = {}
    
    # 验证关键点有效性
    def is_valid(idx):
        return idx < len(keypoint_scores) and keypoint_scores[idx] > 0.5
    
    # 获取有效关键点
    for idx in range(len(keypoint_scores)):
        if is_valid(idx):
            valid_keypoints[idx] = keypoints[idx]
    
    # 获取自定义关键点
    custom_points = {}
    if custom_keypoints is not None and len(custom_keypoints) > 0:
        custom_scores = keypoint_scores[-len(custom_keypoints):]
        for idx, kpt in enumerate(custom_keypoints):
            if idx < len(custom_scores) and custom_scores[idx] > 0.5:
                custom_points[idx] = kpt
    
    # 计算自定义关键点
    new_keypoints = []
    
    # 1. 计算髂前上棘连线中点
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX):
        # 计算髂前上棘连线中点
        iliac_x = (valid_keypoints[LEFT_HIP_IDX][0] + valid_keypoints[RIGHT_HIP_IDX][0]) / 2
        iliac_y = (valid_keypoints[LEFT_HIP_IDX][1] + valid_keypoints[RIGHT_HIP_IDX][1]) / 2
        new_keypoints.append([iliac_x, iliac_y])
    else:
        # 如果无法计算，添加一个无效的点
        new_keypoints.append([-1, -1])
    
    # 2. 计算颈椎中点
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(RIGHT_SHOULDER_IDX) and is_valid(NOSE_IDX):
        # 首先计算左右肩膀的水平中点
        shoulder_mid_x = (valid_keypoints[LEFT_SHOULDER_IDX][0] + valid_keypoints[RIGHT_SHOULDER_IDX][0]) / 2
        shoulder_mid_y = (valid_keypoints[LEFT_SHOULDER_IDX][1] + valid_keypoints[RIGHT_SHOULDER_IDX][1]) / 2
        
        # 然后计算喉结位置 - 在肩膀中点和鼻子之间的位置
        neck_midpoint_x = shoulder_mid_x + (valid_keypoints[NOSE_IDX][0] - shoulder_mid_x) / 3
        neck_midpoint_y = shoulder_mid_y + (valid_keypoints[NOSE_IDX][1] - shoulder_mid_y) / 3
        new_keypoints.append([neck_midpoint_x, neck_midpoint_y])
    else:
        # 如果无法计算，添加一个无效的点
        new_keypoints.append([-1, -1])
    
    # 如果需要返回自定义关键点
    if return_keypoints:
        return new_keypoints
    
    # 1. 计算头前倾角
    if is_valid(NOSE_IDX) and NECK_MIDPOINT_IDX in custom_points:
        results['头前倾角'] = abs(calculate_angle(valid_keypoints[NOSE_IDX], custom_points[NECK_MIDPOINT_IDX]) - 90)
    
    # 2. 计算头侧倾角
    if is_valid(LEFT_EAR_IDX) and is_valid(RIGHT_EAR_IDX):
        ear_angle = abs(calculate_angle(valid_keypoints[LEFT_EAR_IDX], valid_keypoints[RIGHT_EAR_IDX]))
        results['头侧倾角'] = abs(ear_angle - 0 if ear_angle <= 90 else 180 - ear_angle)
    
    # 3. 计算头旋转角
    if is_valid(LEFT_EAR_IDX) and is_valid(RIGHT_EAR_IDX) and is_valid(NOSE_IDX):
        left_dist = calculate_distance(valid_keypoints[NOSE_IDX], valid_keypoints[LEFT_EAR_IDX])
        right_dist = calculate_distance(valid_keypoints[NOSE_IDX], valid_keypoints[RIGHT_EAR_IDX])
        ratio = abs(left_dist - right_dist) / ((left_dist + right_dist) / 2)
        results['头旋转角'] = ratio * 45  # 粗略估计，最大45度
    
    # 4. 计算肩倾斜角
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(RIGHT_SHOULDER_IDX):
        shoulder_angle = abs(calculate_angle(valid_keypoints[LEFT_SHOULDER_IDX], valid_keypoints[RIGHT_SHOULDER_IDX]))
        results['肩倾斜角'] = abs(shoulder_angle - 0 if shoulder_angle <= 90 else 180 - shoulder_angle)
    
    # 5. 计算圆肩角
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(RIGHT_SHOULDER_IDX) and NECK_MIDPOINT_IDX in custom_points:
        # 取肩膀中点
        shoulder_mid_x = (valid_keypoints[LEFT_SHOULDER_IDX][0] + valid_keypoints[RIGHT_SHOULDER_IDX][0]) / 2
        shoulder_mid_y = (valid_keypoints[LEFT_SHOULDER_IDX][1] + valid_keypoints[RIGHT_SHOULDER_IDX][1]) / 2
        shoulder_mid = [shoulder_mid_x, shoulder_mid_y]
        
        # 计算肩膀中点-颈部中点-鼻子的角度
        if is_valid(NOSE_IDX):
            results['圆肩角'] = calculate_angle(shoulder_mid, custom_points[NECK_MIDPOINT_IDX], valid_keypoints[NOSE_IDX])
    
    # 6. 计算背部角
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(LEFT_HIP_IDX) and is_valid(LEFT_KNEE_IDX):
        results['背部角'] = calculate_angle(valid_keypoints[LEFT_SHOULDER_IDX], valid_keypoints[LEFT_HIP_IDX], valid_keypoints[LEFT_KNEE_IDX])
    elif is_valid(RIGHT_SHOULDER_IDX) and is_valid(RIGHT_HIP_IDX) and is_valid(RIGHT_KNEE_IDX):
        results['背部角'] = calculate_angle(valid_keypoints[RIGHT_SHOULDER_IDX], valid_keypoints[RIGHT_HIP_IDX], valid_keypoints[RIGHT_KNEE_IDX])
    
    # 7. 腹部肥胖度（简化估算）
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX) and is_valid(LEFT_SHOULDER_IDX) and is_valid(RIGHT_SHOULDER_IDX):
        hip_width = calculate_distance(valid_keypoints[LEFT_HIP_IDX], valid_keypoints[RIGHT_HIP_IDX])
        shoulder_width = calculate_distance(valid_keypoints[LEFT_SHOULDER_IDX], valid_keypoints[RIGHT_SHOULDER_IDX])
        # 腰围与肩宽的比率作为肥胖度的粗略估计
        results['腹部肥胖度'] = (hip_width / shoulder_width) * 100 - 65  # 调整为百分比
    
    # 8. 计算腰曲度
    if ILIAC_MIDPOINT_IDX in custom_points and NECK_MIDPOINT_IDX in custom_points:
        # 使用自定义的颈椎中点和髂前上棘中点计算腰曲度
        back_angle = calculate_angle(custom_points[NECK_MIDPOINT_IDX], custom_points[ILIAC_MIDPOINT_IDX])
        results['腰曲度'] = abs(back_angle - 90)
    
    # 9. 计算骨盆前倾角
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX) and ILIAC_MIDPOINT_IDX in custom_points:
        hip_midpoint = [(valid_keypoints[LEFT_HIP_IDX][0] + valid_keypoints[RIGHT_HIP_IDX][0]) / 2,
                        (valid_keypoints[LEFT_HIP_IDX][1] + valid_keypoints[RIGHT_HIP_IDX][1]) / 2]
        pelvic_angle = calculate_angle(hip_midpoint, custom_points[ILIAC_MIDPOINT_IDX])
        results['骨盆前倾角'] = pelvic_angle - 90
    
    # 10. 计算侧中位度
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(LEFT_HIP_IDX) and is_valid(LEFT_ANKLE_IDX):
        results['侧中位度'] = 180 - calculate_angle(valid_keypoints[LEFT_SHOULDER_IDX], valid_keypoints[LEFT_HIP_IDX], valid_keypoints[LEFT_ANKLE_IDX])
    elif is_valid(RIGHT_SHOULDER_IDX) and is_valid(RIGHT_HIP_IDX) and is_valid(RIGHT_ANKLE_IDX):
        results['侧中位度'] = 180 - calculate_angle(valid_keypoints[RIGHT_SHOULDER_IDX], valid_keypoints[RIGHT_HIP_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
    
    # 11-12. 计算腿型角度
    if is_valid(LEFT_HIP_IDX) and is_valid(LEFT_KNEE_IDX) and is_valid(LEFT_ANKLE_IDX):
        results['腿型-左腿'] = calculate_angle(valid_keypoints[LEFT_HIP_IDX], valid_keypoints[LEFT_KNEE_IDX], valid_keypoints[LEFT_ANKLE_IDX])
    
    if is_valid(RIGHT_HIP_IDX) and is_valid(RIGHT_KNEE_IDX) and is_valid(RIGHT_ANKLE_IDX):
        results['腿型-右腿'] = calculate_angle(valid_keypoints[RIGHT_HIP_IDX], valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
    
    # 13-14. 膝关节评估角
    if is_valid(LEFT_HIP_IDX) and is_valid(LEFT_KNEE_IDX) and is_valid(LEFT_ANKLE_IDX):
        results['左膝评估角'] = 180 - calculate_angle(valid_keypoints[LEFT_HIP_IDX], valid_keypoints[LEFT_KNEE_IDX], valid_keypoints[LEFT_ANKLE_IDX])
    
    if is_valid(RIGHT_HIP_IDX) and is_valid(RIGHT_KNEE_IDX) and is_valid(RIGHT_ANKLE_IDX):
        results['右膝评估角'] = 180 - calculate_angle(valid_keypoints[RIGHT_HIP_IDX], valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
    
    # 15. 身体倾斜度
    if NECK_MIDPOINT_IDX in custom_points and ILIAC_MIDPOINT_IDX in custom_points:
        body_tilt_angle = calculate_angle(custom_points[NECK_MIDPOINT_IDX], custom_points[ILIAC_MIDPOINT_IDX])
        results['身体倾斜度'] = abs(body_tilt_angle - 90)
    
    # 16. 足八角
    if is_valid(LEFT_ANKLE_IDX) and is_valid(RIGHT_ANKLE_IDX) and is_valid(LEFT_KNEE_IDX) and is_valid(RIGHT_KNEE_IDX):
        left_foot_angle = calculate_angle(valid_keypoints[LEFT_KNEE_IDX], valid_keypoints[LEFT_ANKLE_IDX])
        right_foot_angle = calculate_angle(valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
        feet_angle_diff = abs(left_foot_angle - right_foot_angle)
        results['足八角'] = feet_angle_diff
    
    # 处理结果，保留一位小数
    for key in results:
        if results[key] is not None:
            results[key] = round(results[key], 1)
    
    return results

def display_posture_analysis(frame, results):
    """在命令行显示姿态分析结果，不再在图像上显示"""
    # 正常范围的定义
    normal_ranges = {
        '头前倾角': '0°～5°',
        '头侧倾角': '0°～2°',
        '头旋转角': '0°～5°',
        '肩倾斜角': '0°～2°',
        '圆肩角': '>65°',
        '背部角': '<39°',
        '腹部肥胖度': '0%～35%',
        '腰曲度': '0°～5°',
        '骨盆前倾角': '-7°～7°',
        '侧中位度': '175°～185°',
        '腿型-左腿': '177°～183°',
        '腿型-右腿': '177°～183°',
        '左膝评估角': '175°～185°',
        '右膝评估角': '175°～185°',
        '身体倾斜度': '0°～2°',
        '足八角': '-5°～11°'
    }
    
    # 打印到控制台
    print("\n===== 姿态分析结果 =====")
    print("参数数据\t测量值\t正常范围\t状态")
    
    # 显示结果
    for idx, (key, value) in enumerate(results.items()):
        if value is not None:
            # 检查是否在正常范围内
            normal = True
            range_str = normal_ranges.get(key, "")
            
            if '～' in range_str:
                parts = range_str.replace('°', '').split('～')
                min_val = float(parts[0].replace('%', ''))
                max_val = float(parts[1].replace('%', ''))
                normal = min_val <= value <= max_val
            elif '<' in range_str:
                max_val = float(range_str.replace('<', '').replace('°', '').replace('%', ''))
                normal = value < max_val
            elif '>' in range_str:
                min_val = float(range_str.replace('>', '').replace('°', '').replace('%', ''))
                normal = value > min_val
            
            # 添加单位
            unit = '%' if '肥胖度' in key else '°'
            
            # 打印到控制台
            status = "正常" if normal else "异常"
            print(f"{key}\t{value}{unit}\t{range_str}\t{status}")
    
    # 不修改图像，直接返回原始帧
    return frame

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """处理单张图像，预测关键点并可视化结果。"""

    # 预测边界框
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # 预测关键点
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # 确保图像是RGB格式用于MMPose处理
    img_rgb = img
    if isinstance(img, str):
        img_rgb = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray) and img.shape[-1] == 3:
        # 如果图像是BGR格式(从OpenCV获取的)，转换为RGB
        img_rgb = mmcv.bgr2rgb(img)

    # 过滤关键点数据，只保留鼻子、双耳和身体关键点，并添加自定义关键点
    if (not args.draw_hands) or (not args.draw_face) or args.draw_iliac_midpoint or args.draw_neck_midpoint:
        data_samples = filter_keypoints(data_samples, args)

    # 分析体态并获取测量结果
    posture_results = None
    if hasattr(data_samples, 'pred_instances') and len(data_samples.pred_instances) > 0:
        # 获取第一个检测到的人的关键点
        keypoints = data_samples.pred_instances.keypoints[0]
        keypoint_scores = data_samples.pred_instances.keypoint_scores[0]
        
        # 获取自定义关键点
        custom_kpts = None
        if hasattr(data_samples, 'custom_keypoints') and len(data_samples.custom_keypoints) > 0:
            custom_kpts = data_samples.custom_keypoints[0]
        
        # 分析体态
        posture_results = analyze_body_posture(keypoints, keypoint_scores, custom_kpts)
        
        # 只在命令行显示分析结果
        if args.show_posture_analysis:
            display_posture_analysis(None, posture_results)

    if visualizer is not None:
        # 确保向可视化器传递RGB格式的图像
        visualizer.add_datasample(
            'result',
            img_rgb,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=False,  # 修改为False以避免在这里显示
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)
        
        # 绘制自定义关键点
        if hasattr(data_samples, 'custom_keypoints'):
            # 获取已绘制的图像 (RGB格式)
            image = visualizer.get_image()
            
            # 遍历每个实例
            for i, custom_keypoints in enumerate(data_samples.custom_keypoints):
                # 获取当前实例的置信度
                custom_keypoint_scores = data_samples.custom_keypoint_scores[i]
                
                # 定义关键点颜色 (注意：这里使用RGB格式)
                colors = [(255, 0, 0), (0, 255, 0)]  # 红色和绿色 (RGB格式)
                
                # 检查是否有有效的自定义关键点
                if len(custom_keypoints) > 0:
                    # 遍历自定义关键点
                    for j, (kpt, score) in enumerate(zip(custom_keypoints, custom_keypoint_scores)):
                        # 只绘制置信度高于阈值的关键点
                        if score > args.kpt_thr:
                            # 获取关键点坐标
                            x, y = int(kpt[0]), int(kpt[1])
                            
                            # 绘制关键点
                            cv2.circle(image, (x, y), args.custom_keypoint_radius, colors[j % len(colors)], -1)
                            
                            # 如果显示关键点索引
                            if args.show_kpt_idx:
                                cv2.putText(image, f"C{j}", (x + 5, y + 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j % len(colors)], 1)
                            
                            # 添加连接线
                            if j == 0:  # 髂前上棘中点
                                # 连接到左右髋关节
                                HIP_LEFT_IDX = 11
                                HIP_RIGHT_IDX = 12
                                
                                keypoints = data_samples.pred_instances.keypoints[i]
                                keypoint_scores = data_samples.pred_instances.keypoint_scores[i]
                                
                                if keypoint_scores[HIP_LEFT_IDX] > args.kpt_thr:
                                    hip_left = keypoints[HIP_LEFT_IDX]
                                    x1, y1 = int(hip_left[0]), int(hip_left[1])
                                    cv2.line(image, (x, y), (x1, y1), colors[j % len(colors)], 
                                             args.custom_keypoint_thickness)
                                
                                if keypoint_scores[HIP_RIGHT_IDX] > args.kpt_thr:
                                    hip_right = keypoints[HIP_RIGHT_IDX]
                                    x2, y2 = int(hip_right[0]), int(hip_right[1])
                                    cv2.line(image, (x, y), (x2, y2), colors[j % len(colors)], 
                                             args.custom_keypoint_thickness)
                            
                            elif j == 1:  # 颈椎中点
                                # 连接到鼻子和左右肩膀中点
                                NOSE_IDX = 0
                                LEFT_SHOULDER_IDX = 5
                                RIGHT_SHOULDER_IDX = 6
                                
                                keypoints = data_samples.pred_instances.keypoints[i]
                                keypoint_scores = data_samples.pred_instances.keypoint_scores[i]
                                
                                # 计算肩膀中点
                                if (keypoint_scores[LEFT_SHOULDER_IDX] > args.kpt_thr and
                                    keypoint_scores[RIGHT_SHOULDER_IDX] > args.kpt_thr):
                                    shoulder_mid_x = (keypoints[LEFT_SHOULDER_IDX, 0] + keypoints[RIGHT_SHOULDER_IDX, 0]) / 2
                                    shoulder_mid_y = (keypoints[LEFT_SHOULDER_IDX, 1] + keypoints[RIGHT_SHOULDER_IDX, 1]) / 2
                                    
                                    # 绘制连接到肩膀中点的线
                                    mid_x, mid_y = int(shoulder_mid_x), int(shoulder_mid_y)
                                    cv2.line(image, (x, y), (mid_x, mid_y), colors[j % len(colors)], 
                                             args.custom_keypoint_thickness)
                                
                                # 连接到鼻子
                                if keypoint_scores[NOSE_IDX] > args.kpt_thr:
                                    nose = keypoints[NOSE_IDX]
                                    x2, y2 = int(nose[0]), int(nose[1])
                                    cv2.line(image, (x, y), (x2, y2), colors[j % len(colors)], 
                                             args.custom_keypoint_thickness)
            
            # 更新可视化器的图像
            visualizer.set_image(image)
    
    return data_samples


def main():
    """使用MMDet进行人体检测，并使用RTMPose全身姿态估计模型进行姿态估计。"""
    
    # 使用程序内定义的配置参数
    args = Config()
    
    assert has_mmdet, '请安装mmdet以运行演示。'
    assert args.show or (args.output_root != ''), '必须设置show=True或提供输出路径'

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, 'webcam.mp4')

    # 构建检测器
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # 构建姿态估计器
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # 构建可视化器
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # 数据集元数据从checkpoint加载并传递给模型
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    video_writer = None
    pred_instances_list = []
    frame_idx = 0

    # 用于显示FPS的变量
    start_time = time.time()
    fps_count = 0
    fps_value = 0
    
    # 用于控制姿态分析结果显示的频率
    posture_analysis_interval = 30  # 每30帧显示一次分析结果
    last_analysis_frame = 0

    print("按ESC键退出程序")
    print("开始实时姿态分析...")

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break

        # 用于计算FPS
        if args.fps:
            fps_count += 1
            if time.time() - start_time > 1:
                fps_value = fps_count / (time.time() - start_time)
                fps_count = 0
                start_time = time.time()

        # 姿态估计
        pred_instances = process_one_image(args, frame, detector,
                                          pose_estimator, visualizer,
                                          0.001)

        if args.save_predictions:
            # 保存预测结果
            pred_instances_list.append(
                dict(
                    frame_id=frame_idx,
                    instances=split_instances(pred_instances)))

        # 获取可视化后的帧
        frame_vis = visualizer.get_image()
        
        # 输出视频
        if output_file:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # 可视化图像的大小可能会因热图的存在而变化
                video_writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    25,  # 保存的帧率
                    (frame_vis.shape[1], frame_vis.shape[0]))

            video_writer.write(mmcv.rgb2bgr(frame_vis))
        
        # 显示预测后的视频（而不是原始视频）
        if args.show:
            # 在预测后的帧上显示FPS
            if args.fps:
                # 注意：frame_vis已经是RGB格式，需要转换回BGR用于OpenCV显示
                frame_vis_bgr = mmcv.rgb2bgr(frame_vis.copy())
                cv2.putText(frame_vis_bgr, f"FPS: {fps_value:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('姿态预测结果', frame_vis_bgr)
            else:
                # 确保转换为BGR格式用于OpenCV显示
                cv2.imshow('姿态预测结果', mmcv.rgb2bgr(frame_vis))

        # 显示姿态分析结果 (每隔一定帧数)
        if frame_idx - last_analysis_frame >= posture_analysis_interval:
            last_analysis_frame = frame_idx
            print("\n===== 第 {} 帧分析结果 =====".format(frame_idx))

        # 检查键盘输入
        if cv2.waitKey(1) & 0xFF == 27:  # ESC键
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    if args.save_predictions:
        with open(os.path.join(args.output_root, 'result.json'), 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'结果已保存在 {os.path.join(args.output_root, "result.json")}')


if __name__ == '__main__':
    main() 