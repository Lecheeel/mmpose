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
    
    # 设置输入输出文件
    input_file = 'test.mp4'
    output_file = 'output.mp4'
    
    assert has_mmdet, '请安装mmdet以运行演示。'

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
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)
    video_writer = None
    pred_instances_list = []
    frame_idx = 0

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"处理视频: {input_file}")
    print(f"总帧数: {total_frames}, FPS: {fps}, 分辨率: {width}x{height}")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    progress_step = max(1, total_frames // 100)  # 每处理1%的帧数更新一次进度

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break
            
        # 显示处理进度
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"处理进度: {progress:.1f}%")

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
        if video_writer is None:
            # 可视化图像的大小可能会因热图的存在而变化
            video_writer = cv2.VideoWriter(
                output_file,
                fourcc,
                fps,  # 使用原始视频的帧率
                (frame_vis.shape[1], frame_vis.shape[0]))

        video_writer.write(mmcv.rgb2bgr(frame_vis))

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"处理完成，输出文件: {output_file}")

    if args.save_predictions:
        result_file = 'result.json'
        with open(result_file, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'结果已保存在 {result_file}')


if __name__ == '__main__':
    main() 