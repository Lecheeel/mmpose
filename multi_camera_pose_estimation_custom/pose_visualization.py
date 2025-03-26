import cv2
import numpy as np
from typing import Dict, List
from . import config

def process_pose_results(frame: np.ndarray, results: List, call_args: Dict) -> np.ndarray:
    """处理姿态估计结果并绘制到帧上
    
    Args:
        frame: 原始视频帧
        results: 姿态估计结果
        call_args: 调用参数
        
    Returns:
        处理后的帧
    """
    display_frame = frame.copy()
    
    # 定义需要保留的关键点索引 (0-16: 身体关键点, 17-22: 足部关键点)
    kept_indices = list(range(0, 23))  # 0-22的索引
    
    # 定义骨架连接关系 (关键点索引对)，使用MMPose连接样式
    skeleton = [
        # 头部连接
        (0, 1), (0, 2),         # 鼻子到左/右眼
        (1, 3), (2, 4),         # 左/右眼到左/右耳
        # 上身躯干
        (0, 5), (0, 6), (5, 6), # 颈部三角形
        (5, 7), (7, 9),         # 左肩到左手腕
        (6, 8), (8, 10),        # 右肩到右手腕
        # 下半身
        (5, 11), (6, 12),       # 肩膀到髋部
        (11, 12),               # 左右髋部连接
        (11, 13), (13, 15),     # 左髋到左脚踝
        (12, 14), (14, 16),     # 右髋到右脚踝
        # 足部连接
        (15, 19), (15, 17),  
        (15, 18), (16, 22), 
        (16, 21), (16, 20)
    ]
    
    # 为骨架连接定义颜色 - 使用配置中的颜色
    link_colors = config.ColorConfig.SKELETON_COLORS

    # 人员计数器
    person_count = 0

    # 遍历结果
    try:
        for result in results:
            pred_instances = result.get('predictions', [])
            
            # 如果有预测结果，在原始帧上绘制
            if pred_instances and len(pred_instances) > 0:
                # 处理每个预测实例
                for instance_list in pred_instances:
                    # 检查是否为列表类型（多个人的情况）
                    if isinstance(instance_list, list):
                        for instance in instance_list:
                            # 绘制单个人的姿态
                            process_single_person(display_frame, instance, person_count, kept_indices, 
                                                 skeleton, link_colors, call_args)
                            person_count += 1
                    # 检查是否为字典类型（单个人的情况）
                    elif isinstance(instance_list, dict):
                        instance = instance_list
                        process_single_person(display_frame, instance, person_count, kept_indices, 
                                             skeleton, link_colors, call_args)
                        person_count += 1
    except Exception as e:
        print(f"处理姿态结果绘制时出错: {str(e)}")
    
    return display_frame

def process_single_person(display_frame, instance, person_idx, kept_indices, skeleton, link_colors, call_args):
    """处理单个人的姿态估计结果
    
    Args:
        display_frame: 显示帧
        instance: 单个人的姿态估计结果
        person_idx: 人员索引
        kept_indices: 保留的关键点索引
        skeleton: 骨架连接关系
        link_colors: 连接线颜色
        call_args: 调用参数
    """
    try:
        # 获取关键点和得分
        keypoints = instance.get('keypoints', None)
        keypoint_scores = instance.get('keypoint_scores', None)
        
        # 获取track_id（如果存在）或使用序号作为ID
        track_id = instance.get('track_id', person_idx)
        
        if keypoints is not None and keypoint_scores is not None:
            # 确保为numpy数组并检查是否为空
            keypoints = np.array(keypoints)
            keypoint_scores = np.array(keypoint_scores)
            
            if keypoints.size == 0 or keypoint_scores.size == 0:
                return  # 如果关键点为空，直接返回
            
            # 检查关键点数量是否足够
            if len(keypoints) >= max(kept_indices) + 1:
                # 计算新的关键点: 颈椎中点（鼻子和左右肩中点之间的点）
                neck_valid = (keypoint_scores[0] > call_args['kpt_thr'] and 
                             keypoint_scores[5] > call_args['kpt_thr'] and 
                             keypoint_scores[6] > call_args['kpt_thr'])
                
                if neck_valid:
                    # 计算左右肩中点
                    shoulder_mid_x = (keypoints[5][0] + keypoints[6][0]) / 2
                    shoulder_mid_y = (keypoints[5][1] + keypoints[6][1]) / 2
                    
                    # 计算颈椎中点（鼻子和肩膀中点之间的某个位置，这里取1/3处）
                    neck_vertebra_x = int(keypoints[0][0] * 0.3 + shoulder_mid_x * 0.7)
                    neck_vertebra_y = int(keypoints[0][1] * 0.3 + shoulder_mid_y * 0.7)
                    neck_vertebra_point = (neck_vertebra_x, neck_vertebra_y)
                    
                    # 绘制颈椎中点
                    cv2.circle(display_frame, neck_vertebra_point, call_args['radius'], config.ColorConfig.NECK_VERTEBRA_COLOR, -1)
                
                # 计算新的关键点: 髂前上棘连线中点（左右髋部的中点）
                hip_valid = (keypoint_scores[11] > call_args['kpt_thr'] and 
                            keypoint_scores[12] > call_args['kpt_thr'])
                
                if hip_valid:
                    hip_mid_x = int((keypoints[11][0] + keypoints[12][0]) / 2)
                    hip_mid_y = int((keypoints[11][1] + keypoints[12][1]) / 2)
                    hip_mid_point = (hip_mid_x, hip_mid_y)
                    
                    # 绘制髂前上棘连线中点
                    cv2.circle(display_frame, hip_mid_point, call_args['radius'], config.ColorConfig.HIP_MID_COLOR, -1)
                
                # 如果两个点都有效，绘制它们之间的连线
                if neck_valid and hip_valid:
                    cv2.line(display_frame, neck_vertebra_point, hip_mid_point, config.ColorConfig.SPINE_COLOR, call_args['thickness'])
                
                # 只绘制需要的关键点
                for idx in kept_indices:
                    if idx < len(keypoints) and keypoint_scores[idx] > call_args['kpt_thr']:
                        x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                        cv2.circle(display_frame, (x, y), call_args['radius'], config.ColorConfig.KEYPOINT_COLOR, -1)
                
                # 绘制骨架连线
                for sk_idx, (start_idx, end_idx) in enumerate(skeleton):
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                        keypoint_scores[start_idx] > call_args['kpt_thr'] and 
                        keypoint_scores[end_idx] > call_args['kpt_thr']):
                        
                        start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        
                        color = link_colors[sk_idx] if sk_idx < len(link_colors) else config.ColorConfig.KEYPOINT_COLOR
                        thickness = call_args['thickness']
                        cv2.line(display_frame, start_pt, end_pt, color, thickness)
                
                # 在头部上方绘制ID标签
                if keypoint_scores[0] > call_args['kpt_thr']:  # 如果鼻子关键点可见
                    id_x = int(keypoints[0][0])
                    id_y = int(keypoints[0][1] - 30)  # 在头部上方放置ID标签
                    # 绘制黑色背景框增强可读性
                    padx, pady = config.DisplayConfig.ID_BACKGROUND_PADDING
                    cv2.rectangle(display_frame, (id_x-padx, id_y-pady), (id_x+padx, id_y+5), (0, 0, 0), -1)
                    # 绘制ID文本
                    dx, dy = config.DisplayConfig.ID_TEXT_OFFSET
                    cv2.putText(display_frame, f"ID:{track_id}", (id_x+dx, id_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, config.DisplayConfig.ID_TEXT_SCALE, 
                               config.DisplayConfig.ID_TEXT_COLOR, config.DisplayConfig.ID_TEXT_THICKNESS)
            else:
                # 如果点数不够，只绘制可用关键点
                for kpt_idx, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                    if kpt_idx in kept_indices and score > call_args['kpt_thr']:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(display_frame, (x, y), call_args['radius'], config.ColorConfig.KEYPOINT_COLOR, -1)
    except Exception as e:
        print(f"绘制单个人姿态时出错: {str(e)}")
        return
    
    # 检索边界框（如果存在）
    try:
        bbox = instance.get('bbox', None)
        if bbox is not None and call_args['draw_bbox']:
            # 处理不同格式的边界框
            if isinstance(bbox, list) or isinstance(bbox, np.ndarray):
                if len(bbox) == 4:
                    # 标准格式 [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(float(coord)) if isinstance(coord, (int, float, str)) else int(coord[0]) for coord in bbox]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), config.ColorConfig.BBOX_COLOR, config.ColorConfig.BBOX_THICKNESS)
                    # 在边界框左上角绘制ID标签
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, config.DisplayConfig.ID_TEXT_SCALE, 
                               config.ColorConfig.BBOX_ID_COLOR, config.DisplayConfig.ID_TEXT_THICKNESS)
                elif len(bbox) == 5:
                    # 带置信度的格式 [x1, y1, x2, y2, score]
                    x1, y1, x2, y2, _ = [int(float(coord)) if isinstance(coord, (int, float, str)) else int(coord[0]) for coord in bbox[:5]]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), config.ColorConfig.BBOX_COLOR, config.ColorConfig.BBOX_THICKNESS)
                    # 在边界框左上角绘制ID标签
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, config.DisplayConfig.ID_TEXT_SCALE, 
                               config.ColorConfig.BBOX_ID_COLOR, config.DisplayConfig.ID_TEXT_THICKNESS)
    except Exception as e:
        print(f"处理边界框时出错: {str(e)}, 边界框数据: {bbox}") 