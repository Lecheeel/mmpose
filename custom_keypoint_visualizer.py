import cv2
import numpy as np
import math
import os
import argparse
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector

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

def compute_custom_keypoints(keypoints, keypoint_scores, thr=0.5):
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
        # 在足踝外侧估算第五跖骨
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
            # 根据膝盖到踝关节的方向估算足跟位置
            angle = calculate_angle(valid_keypoints[LEFT_KNEE_IDX], valid_keypoints[LEFT_ANKLE_IDX])
            dist = 20  # 距离足踝的像素距离
            x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle+180))
            y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle+180))
        else:
            # 简单地在足踝后方估算
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

def draw_custom_keypoints(image, custom_points, color_map=None, radius=5, thickness=2):
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
        # 默认颜色映射
        color_map = {
            'N1': (255, 0, 0),      # 乳突 - 蓝色
            'N2_left': (0, 255, 0), # 肩峰 - 绿色
            'N2_right': (0, 255, 0),
            'N3': (0, 0, 255),      # 第一胸椎 - 红色
            'N4': (255, 255, 0),    # 第十二胸椎 - 青色
            'N5': (255, 0, 255),    # 第一腰椎 - 洋红色
            'N6': (0, 255, 255),    # 第五腰椎 - 黄色
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
            'N13': (128, 128, 128),     # 脊柱棘突 - 灰色
            'N14_left': (230, 230, 250), # 双侧髂嵴 - 淡紫色
            'N14_right': (230, 230, 250),
            'N15': (255, 250, 205),     # 髂后上嵴 - 米色
            'N16_left': (173, 216, 230), # 第一跖骨 - 淡蓝色
            'N16_right': (173, 216, 230),
            'N17_left': (152, 251, 152), # 足尖 - 淡绿色
            'N17_right': (152, 251, 152),
            'neck_mid': (255, 99, 71)   # 颈椎中点 - 番茄色
        }
    
    # 绘制所有关键点
    for key, point in custom_points.items():
        color = color_map.get(key, (0, 0, 255))
        
        if key == 'N13' and isinstance(point, list):
            # 脊柱棘突是多个点，连接它们
            for i in range(len(point) - 1):
                cv2.line(image, point[i], point[i+1], color, thickness)
                cv2.circle(image, point[i], radius, color, -1)
            if point:
                cv2.circle(image, point[-1], radius, color, -1)
        else:
            cv2.circle(image, point, radius, color, -1)
            
            # 标注关键点
            if not key.endswith('_left') and not key.endswith('_right'):
                cv2.putText(image, key, (point[0] + 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # 对于左右侧的关键点，只显示基本名称
                base_key = key.split('_')[0]
                cv2.putText(image, base_key, (point[0] + 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 绘制连接线以显示脊柱曲线
    spine_keys = ['N3', 'N4', 'N5', 'N6', 'N8']
    spine_points = []
    
    for key in spine_keys:
        if key in custom_points:
            spine_points.append(custom_points[key])
    
    # 连接脊柱关键点
    for i in range(len(spine_points) - 1):
        cv2.line(image, spine_points[i], spine_points[i+1], (128, 128, 128), thickness)
    
    return image

def process_image(image_path, det_model, pose_model, save_path=None, visualize=True):
    """处理单张图像，检测人体关键点并绘制所有需要的自定义关键点
    
    Args:
        image_path: 输入图像路径
        det_model: 人体检测模型
        pose_model: 姿态估计模型
        save_path: 保存结果的路径，为None则不保存
        visualize: 是否显示结果图像
    
    Returns:
        result_image: 绘制了所有关键点的图像
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return None
    
    # 人体检测
    det_results = inference_detector(det_model, image)
    pred_instances = det_results.pred_instances
    
    if len(pred_instances.bboxes) == 0:
        print("没有检测到人体")
        return image
    
    # 获取置信度最高的人体边界框
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    best_idx = np.argmax(scores)
    person_bbox = bboxes[best_idx]
    
    # 姿态估计
    pose_results = inference_topdown(pose_model, image, [{"bbox": person_bbox}])
    
    if not pose_results or len(pose_results) == 0:
        print("姿态估计失败")
        return image
    
    # 获取关键点
    keypoints = pose_results[0].pred_instances.keypoints[0].cpu().numpy()
    keypoint_scores = pose_results[0].pred_instances.keypoint_scores[0].cpu().numpy()
    
    # 计算所有自定义关键点
    custom_points = compute_custom_keypoints(keypoints, keypoint_scores)
    
    # 绘制原始关键点
    result_image = image.copy()
    for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
        if score > 0.5:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(result_image, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(result_image, str(i), (x + 3, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制自定义关键点
    result_image = draw_custom_keypoints(result_image, custom_points)
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, result_image)
    
    # 显示结果
    if visualize:
        cv2.imshow('Custom Keypoints', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制用于体态评估的自定义关键点')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
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
    parser.add_argument('--save-path', type=str, default=None, help='保存结果图像的路径')
    parser.add_argument('--no-visualize', action='store_true', help='不显示结果')
    
    args = parser.parse_args()
    
    # 初始化模型
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    
    # 处理图像
    process_image(args.image, det_model, pose_model, 
                  save_path=args.save_path, 
                  visualize=not args.no_visualize)

if __name__ == '__main__':
    main() 