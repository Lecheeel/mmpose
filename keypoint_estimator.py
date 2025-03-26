import cv2
import torch
import numpy as np
import sys
import os
import time
import argparse
import math
from mmpose.apis import inference_topdown, init_model
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description='关键点估算与标注工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备编号')
    parser.add_argument('--output_dir', type=str, default='output_images', help='输出图像保存目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备，如cuda:0或cpu')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--save_frames', action='store_true', help='是否保存处理后的视频帧')
    return parser.parse_args()

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

def compute_keypoints(keypoints, keypoint_scores, thr=0.3):
    """计算和估计关键点坐标
    
    Args:
        keypoints: 关键点坐标数组，形状为(N, 2)
        keypoint_scores: 关键点置信度分数，形状为(N,)
        thr: 关键点有效性阈值
    
    Returns:
        dict: 包含所有关键点的字典
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
    
    # 所有关键点字典
    all_points = {}
    
    # 先存储原始关键点
    for idx in valid_keypoints:
        point_name = f"原始_{idx}"
        all_points[point_name] = (int(valid_keypoints[idx][0]), int(valid_keypoints[idx][1]))
    
    # 计算肩膀中点(用于后续计算)
    shoulder_mid = None
    if is_valid(LEFT_SHOULDER_IDX) and is_valid(RIGHT_SHOULDER_IDX):
        mid_x = (valid_keypoints[LEFT_SHOULDER_IDX][0] + valid_keypoints[RIGHT_SHOULDER_IDX][0]) / 2
        mid_y = (valid_keypoints[LEFT_SHOULDER_IDX][1] + valid_keypoints[RIGHT_SHOULDER_IDX][1]) / 2
        shoulder_mid = (int(mid_x), int(mid_y))
        all_points["肩膀中点"] = shoulder_mid
    
    # 计算髋部中点(用于后续计算)
    hip_mid = None
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX):
        mid_x = (valid_keypoints[LEFT_HIP_IDX][0] + valid_keypoints[RIGHT_HIP_IDX][0]) / 2
        mid_y = (valid_keypoints[LEFT_HIP_IDX][1] + valid_keypoints[RIGHT_HIP_IDX][1]) / 2
        hip_mid = (int(mid_x), int(mid_y))
        all_points["N8"] = hip_mid  # N8: 髂前上棘
    
    # N1: 左右乳突(通过耳朵位置偏移计算)
    if is_valid(LEFT_EAR_IDX):
        # 左乳突: 耳朵位置向后下方偏移
        offset_x = -10  # 向后偏移10像素
        offset_y = 15   # 向下偏移15像素
        x = valid_keypoints[LEFT_EAR_IDX][0] + offset_x
        y = valid_keypoints[LEFT_EAR_IDX][1] + offset_y
        all_points['N1_左'] = (int(x), int(y))
        
    if is_valid(RIGHT_EAR_IDX):
        # 右乳突: 耳朵位置向后下方偏移
        offset_x = 10   # 向后偏移10像素
        offset_y = 15   # 向下偏移15像素
        x = valid_keypoints[RIGHT_EAR_IDX][0] + offset_x
        y = valid_keypoints[RIGHT_EAR_IDX][1] + offset_y
        all_points['N1_右'] = (int(x), int(y))
    
    # N2: 肩峰(左右肩膀)
    if is_valid(LEFT_SHOULDER_IDX):
        all_points['N2_左'] = (int(valid_keypoints[LEFT_SHOULDER_IDX][0]), 
                             int(valid_keypoints[LEFT_SHOULDER_IDX][1]))
    if is_valid(RIGHT_SHOULDER_IDX):
        all_points['N2_右'] = (int(valid_keypoints[RIGHT_SHOULDER_IDX][0]), 
                             int(valid_keypoints[RIGHT_SHOULDER_IDX][1]))
    
    # N3: 第一胸椎(颈部下方)
    if shoulder_mid and is_valid(NOSE_IDX):
        # 在肩膀中点和鼻子之间取一点作为颈椎中点
        neck_x = shoulder_mid[0] * 0.7 + valid_keypoints[NOSE_IDX][0] * 0.3
        neck_y = shoulder_mid[1] * 0.8 + valid_keypoints[NOSE_IDX][1] * 0.2
        all_points['颈椎中点'] = (int(neck_x), int(neck_y))
        
        # 第一胸椎在颈椎中点下方
        x = neck_x
        y = neck_y + (shoulder_mid[1] - neck_y) * 0.3
        all_points['N3'] = (int(x), int(y))
    
    # 如果有肩膀中点和髋部中点，计算脊柱上的点
    if shoulder_mid and hip_mid:
        # 计算脊柱线(从肩膀中点到髋部中点)
        spine_length = calculate_distance(shoulder_mid, hip_mid)
        spine_angle = calculate_angle(shoulder_mid, hip_mid)
        
        # N4: 第十二胸椎(脊柱上点)
        ratio = 0.6  # 大约位于肩膀和髋部之间60%处
        x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
        y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
        all_points['N4'] = (int(x), int(y))
        
        # N5: 第一腰椎(脊柱上点)
        ratio = 0.7  # 大约位于肩膀和髋部之间70%处
        x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
        y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
        all_points['N5'] = (int(x), int(y))
        
        # N6: 第五腰椎(髋部中点上方)
        ratio = 0.9  # 大约位于肩膀和髋部之间90%处
        x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
        y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
        all_points['N6'] = (int(x), int(y))
        
        # N13: 脊柱棘突(沿脊柱线的多个点)
        spine_points = []
        for i in range(5):
            ratio = 0.2 + i * 0.15  # 均匀分布在脊柱上
            x = shoulder_mid[0] + ratio * (hip_mid[0] - shoulder_mid[0])
            y = shoulder_mid[1] + ratio * (hip_mid[1] - shoulder_mid[1])
            spine_points.append((int(x), int(y)))
        all_points['N13'] = spine_points
    
    # N7: 耻骨联合(髋部中点下方)
    if hip_mid:
        # 耻骨联合在髋部下方一定距离
        x = hip_mid[0]
        y = hip_mid[1] + (hip_mid[1] - shoulder_mid[1]) * 0.15 if shoulder_mid else hip_mid[1] + 20
        all_points['N7'] = (int(x), int(y))
    
    # N9: 髌骨(膝关节)
    if is_valid(LEFT_KNEE_IDX):
        all_points['N9_左'] = (int(valid_keypoints[LEFT_KNEE_IDX][0]), 
                             int(valid_keypoints[LEFT_KNEE_IDX][1]))
    if is_valid(RIGHT_KNEE_IDX):
        all_points['N9_右'] = (int(valid_keypoints[RIGHT_KNEE_IDX][0]), 
                             int(valid_keypoints[RIGHT_KNEE_IDX][1]))
    
    # N10: 第五跖骨(足踝外侧)
    if is_valid(LEFT_ANKLE_IDX):
        angle = 45 if is_valid(LEFT_KNEE_IDX) else 0
        dist = 15  # 距离足踝的像素距离
        x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        all_points['N10_左'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        angle = -45 if is_valid(RIGHT_KNEE_IDX) else 0
        dist = 15
        x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        all_points['N10_右'] = (int(x), int(y))
    
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
        all_points['N11_左'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        if is_valid(RIGHT_KNEE_IDX):
            angle = calculate_angle(valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
            dist = 20
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle+180))
            y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle+180))
        else:
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] - 20
            y = valid_keypoints[RIGHT_ANKLE_IDX][1]
        all_points['N11_右'] = (int(x), int(y))
    
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
        all_points['N12_左'] = (int(x), int(y))
    
    if is_valid(RIGHT_WRIST_IDX):
        if is_valid(RIGHT_ELBOW_IDX):
            angle = calculate_angle(valid_keypoints[RIGHT_ELBOW_IDX], valid_keypoints[RIGHT_WRIST_IDX])
            dist = 25
            x = valid_keypoints[RIGHT_WRIST_IDX][0] + dist * math.cos(math.radians(angle))
            y = valid_keypoints[RIGHT_WRIST_IDX][1] + dist * math.sin(math.radians(angle))
        else:
            x = valid_keypoints[RIGHT_WRIST_IDX][0] + 25
            y = valid_keypoints[RIGHT_WRIST_IDX][1]
        all_points['N12_右'] = (int(x), int(y))
    
    # N14: 双侧髂嵴(髋部两侧)
    if is_valid(LEFT_HIP_IDX) and is_valid(RIGHT_HIP_IDX):
        hip_center_x = (valid_keypoints[LEFT_HIP_IDX][0] + valid_keypoints[RIGHT_HIP_IDX][0]) / 2
        hip_center_y = (valid_keypoints[LEFT_HIP_IDX][1] + valid_keypoints[RIGHT_HIP_IDX][1]) / 2
        hip_width = calculate_distance(valid_keypoints[LEFT_HIP_IDX], valid_keypoints[RIGHT_HIP_IDX])
        
        left_x = valid_keypoints[LEFT_HIP_IDX][0] - hip_width * 0.15
        left_y = valid_keypoints[LEFT_HIP_IDX][1]
        right_x = valid_keypoints[RIGHT_HIP_IDX][0] + hip_width * 0.15
        right_y = valid_keypoints[RIGHT_HIP_IDX][1]
        
        all_points['N14_左'] = (int(left_x), int(left_y))
        all_points['N14_右'] = (int(right_x), int(right_y))
    
    # N15: 髂后上嵴(髋部后方)
    if hip_mid:
        # 髂后上嵴位于髋部中点后方
        back_dist = 30  # 向后的距离
        x = hip_mid[0] - back_dist
        y = hip_mid[1]
        all_points['N15'] = (int(x), int(y))
    
    # N16: 第一跖骨(足踝内侧)
    if is_valid(LEFT_ANKLE_IDX):
        angle = -45 if is_valid(LEFT_KNEE_IDX) else 0
        dist = 15
        x = valid_keypoints[LEFT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[LEFT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        all_points['N16_左'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        angle = 45 if is_valid(RIGHT_KNEE_IDX) else 0
        dist = 15
        x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
        y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        all_points['N16_右'] = (int(x), int(y))
    
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
        all_points['N17_左'] = (int(x), int(y))
    
    if is_valid(RIGHT_ANKLE_IDX):
        if is_valid(RIGHT_KNEE_IDX):
            angle = calculate_angle(valid_keypoints[RIGHT_KNEE_IDX], valid_keypoints[RIGHT_ANKLE_IDX])
            dist = 40
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] + dist * math.cos(math.radians(angle))
            y = valid_keypoints[RIGHT_ANKLE_IDX][1] + dist * math.sin(math.radians(angle))
        else:
            x = valid_keypoints[RIGHT_ANKLE_IDX][0] + 40
            y = valid_keypoints[RIGHT_ANKLE_IDX][1]
        all_points['N17_右'] = (int(x), int(y))
    
    return all_points

def get_chinese_font():
    """获取中文字体"""
    try:
        # 尝试使用系统中文字体
        if sys.platform.startswith('win'):
            font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows下的黑体
        elif sys.platform.startswith('darwin'):
            font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS
        else:
            font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux
            
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, 32)
    except Exception as e:
        print(f"加载中文字体失败: {e}")
    return None

def put_chinese_text(img, text, position, text_color=(0, 255, 0)):
    """在图片上添加中文文字"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font()
    if font:
        draw.text(position, text, font=font, fill=text_color)
        return np.array(img_pil)
    else:
        # 如果无法加载中文字体，使用OpenCV默认字体
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        return img

def draw_keypoints(image, all_points, display_original=False):
    """绘制关键点和连接线"""
    # 定义颜色映射
    color_map = {
        'N1': (0, 0, 255),      # 乳突 - 红色
        'N2': (0, 255, 0),      # 肩峰 - 绿色
        'N3': (255, 0, 0),      # 第一胸椎 - 蓝色
        'N4': (0, 255, 255),    # 第十二胸椎 - 黄色
        'N5': (255, 0, 255),    # 第一腰椎 - 洋红色
        'N6': (255, 255, 0),    # 第五腰椎 - 青色
        'N7': (128, 0, 128),    # 耻骨联合 - 紫色
        'N8': (255, 165, 0),    # 髂前上棘 - 橙色
        'N9': (255, 192, 203),  # 髌骨 - 粉色
        'N10': (0, 128, 0),     # 第五跖骨 - 深绿色
        'N11': (139, 69, 19),   # 足跟 - 棕色
        'N12': (70, 130, 180),  # 手掌 - 钢蓝色
        'N13': (255, 255, 255), # 脊柱棘突 - 白色
        'N14': (230, 230, 250), # 双侧髂嵴 - 淡紫色
        'N15': (255, 250, 205), # 髂后上嵴 - 米色
        'N16': (173, 216, 230), # 第一跖骨 - 淡蓝色
        'N17': (152, 251, 152), # 足尖 - 淡绿色
    }
    
    # 点计数
    point_count = 0
    
    # 先绘制原始关键点（如果需要）
    if display_original:
        for key, point in all_points.items():
            if key.startswith('原始_'):
                cv2.circle(image, point, 3, (100, 100, 100), -1)
                image = put_chinese_text(image, key, (point[0] + 3, point[1] - 3), (100, 100, 100))
    
    # 绘制其他关键点
    for key, point in all_points.items():
        # 跳过原始关键点和辅助点
        if key.startswith('原始_') or key in ['肩膀中点', '颈椎中点']:
            continue
            
        # 获取基本关键点名称（去掉左右后缀）
        base_key = key.split('_')[0]
        color = color_map.get(base_key, (0, 0, 255))
        
        if key == 'N13' and isinstance(point, list):
            # 脊柱棘突是多个点，连接它们
            for i in range(len(point) - 1):
                cv2.line(image, point[i], point[i+1], color, 2)
                cv2.circle(image, point[i], 5, color, -1)
                point_count += 1
            if point:
                cv2.circle(image, point[-1], 5, color, -1)
                point_count += 1
            # 只标注一次
            mid_idx = len(point) // 2
            if mid_idx < len(point):
                mid_point = point[mid_idx]
                image = put_chinese_text(image, base_key, (mid_point[0] + 5, mid_point[1] - 5), color)
        else:
            cv2.circle(image, point, 5, color, -1)
            point_count += 1
            
            # 标注关键点
            if not key.endswith('_左') and not key.endswith('_右'):
                image = put_chinese_text(image, key, (point[0] + 5, point[1] - 5), color)
            else:
                # 对于左右侧的关键点，显示基本名称和左右标识
                side = key.split('_')[-1]
                label = f"{base_key}{side}"
                image = put_chinese_text(image, label, (point[0] + 5, point[1] - 5), color)
    
    # 绘制脊柱连接线
    spine_keys = ['N3', 'N4', 'N5', 'N6', 'N8']
    spine_points = []
    
    for key in spine_keys:
        if key in all_points:
            spine_points.append(all_points[key])
    
    # 连接脊柱关键点
    for i in range(len(spine_points) - 1):
        cv2.line(image, spine_points[i], spine_points[i+1], (255, 255, 255), 2)
    
    # 添加关键点数量信息
    image = put_chinese_text(image, f"关键点数量: {point_count}", (10, 30), (0, 255, 0))
    
    return image

def main():
    args = parse_args()
    debug = args.debug
    
    # 检查CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出文件夹
    if args.save_frames and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载姿态估计模型
    print("加载模型中...")
    # 使用RTMPose-L模型
    # 尝试多种可能的模型配置路径
    try:
        # 尝试路径1
        model_config = "configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
        model_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
        
        pose_model = init_model(model_config, model_checkpoint, device=device)
        print(f"使用配置路径: {model_config}")
    except Exception as e1:
        print(f"加载模型配置1失败: {e1}")
        try:
            # 尝试路径2
            model_config = "projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
            model_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
            
            pose_model = init_model(model_config, model_checkpoint, device=device)
            print(f"使用配置路径: {model_config}")
        except Exception as e2:
            print(f"加载模型配置2失败: {e2}")
            try:
                # 尝试路径3 - 最简单的方式，直接使用mmpose提供的预定义配置
                from mmpose.apis import init_detector
                
                pose_model = init_detector(device=device)
                print("使用MMPose默认模型配置")
            except Exception as e3:
                print(f"所有模型配置尝试均失败")
                import traceback
                traceback.print_exc()
                return
                
    print("模型加载成功！")
    
    # 打开摄像头
    print(f"正在打开摄像头 {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # 使用DirectShow后端
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 使用MJPG编码
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小为1，减少延迟
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
    
    # 获取摄像头分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头分辨率: {width}x{height}, FPS: {fps}")
    
    frame_count = 0
    start_time = time.time()
    
    print("摄像头已打开，按'q'键退出...")
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break
            
            frame_time = time.time()
            frame_count += 1
            
            # 创建显示帧
            display_frame = frame.copy()
            
            try:
                # 姿态估计
                pose_results = inference_topdown(pose_model, frame)
                
                if pose_results:
                    # 获取第一个人的关键点
                    result = pose_results[0]
                    
                    # 直接从结果中提取关键点和分数
                    try:
                        # 检查结果格式，适配不同版本的mmpose
                        if hasattr(result, 'pred_instances'):
                            # 新版MMPose格式
                            if hasattr(result.pred_instances, 'keypoints') and len(result.pred_instances.keypoints) > 0:
                                # 检查是否是PyTorch张量还是numpy数组
                                if hasattr(result.pred_instances.keypoints[0], 'cpu'):
                                    keypoints = result.pred_instances.keypoints[0].cpu().numpy()
                                else:
                                    keypoints = result.pred_instances.keypoints[0]
                                    
                                if hasattr(result.pred_instances.keypoint_scores[0], 'cpu'):
                                    keypoint_scores = result.pred_instances.keypoint_scores[0].cpu().numpy()
                                else:
                                    keypoint_scores = result.pred_instances.keypoint_scores[0]
                            else:
                                raise ValueError("未找到有效的关键点数据")
                        elif hasattr(result, 'keypoints') or hasattr(result, 'keypoint_scores'):
                            # 可能是旧版MMPose或其他格式
                            if hasattr(result, 'keypoints'):
                                keypoints = result.keypoints[0].cpu().numpy() if hasattr(result.keypoints[0], 'cpu') else result.keypoints[0]
                            elif 'keypoints' in result:
                                keypoints = result['keypoints'][0].cpu().numpy() if hasattr(result['keypoints'][0], 'cpu') else result['keypoints'][0]
                            else:
                                raise ValueError("未找到关键点数据")
                                
                            if hasattr(result, 'keypoint_scores'):
                                keypoint_scores = result.keypoint_scores[0].cpu().numpy() if hasattr(result.keypoint_scores[0], 'cpu') else result.keypoint_scores[0]
                            elif 'keypoint_scores' in result:
                                keypoint_scores = result['keypoint_scores'][0].cpu().numpy() if hasattr(result['keypoint_scores'][0], 'cpu') else result['keypoint_scores'][0]
                            else:
                                raise ValueError("未找到关键点分数数据")
                        else:
                            # 尝试直接处理字典结构
                            if isinstance(result, dict):
                                for key in ['keypoints', 'kpts', 'pose', 'keypoint', 'joints']:
                                    if key in result and len(result[key]) > 0:
                                        keypoints = result[key][0]
                                        break
                                else:
                                    raise ValueError("在字典结构中未找到关键点数据")
                                    
                                for key in ['keypoint_scores', 'scores', 'score', 'kpt_scores']:
                                    if key in result and len(result[key]) > 0:
                                        keypoint_scores = result[key][0]
                                        break
                                else:
                                    # 如果没有找到分数，创建默认分数
                                    print("警告: 未找到关键点分数，使用默认值")
                                    keypoint_scores = np.ones(len(keypoints)) * 0.9
                            else:
                                raise ValueError("无法识别的结果格式")
                        
                        # 计算所有关键点
                        all_points = compute_keypoints(keypoints, keypoint_scores, thr=0.3)
                        
                        # 绘制关键点
                        display_frame = draw_keypoints(display_frame, all_points, display_original=debug)
                        
                        # 计算FPS
                        process_time = time.time() - frame_time
                        current_fps = 1.0 / process_time if process_time > 0 else 0
                        
                        # 显示处理时间和FPS
                        display_frame = put_chinese_text(display_frame, f"处理时间: {process_time:.3f}秒", (10, 60), (0, 255, 0))
                        display_frame = put_chinese_text(display_frame, f"FPS: {current_fps:.1f}", (10, 90), (0, 255, 0))
                    except Exception as e:
                        print(f"处理关键点时出错: {e}")
                        display_frame = put_chinese_text(display_frame, f"处理错误: {str(e)}", (10, 30), (0, 0, 255))
                        if debug:
                            import traceback
                            traceback.print_exc()
                else:
                    display_frame = put_chinese_text(display_frame, "未检测到人体", (10, 30), (0, 0, 255))
                    
            except Exception as e:
                print(f"处理关键点时出错: {e}")
                display_frame = put_chinese_text(display_frame, f"处理错误: {str(e)}", (10, 30), (0, 0, 255))
                if debug:
                    import traceback
                    traceback.print_exc()
            
            # 显示结果
            cv2.imshow('实时关键点估算', display_frame)
            
            # 保存帧（如果需要）
            if args.save_frames:
                output_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, display_frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"程序运行出错: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    finally:
        # 释放摄像头和关闭窗口
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示总体统计信息
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"总运行时间: {total_time:.2f}秒")
        print(f"总帧数: {frame_count}")
        print(f"平均FPS: {avg_fps:.2f}")
        
        if args.save_frames:
            print(f"已保存帧到目录: {args.output_dir}")

if __name__ == '__main__':
    main() 