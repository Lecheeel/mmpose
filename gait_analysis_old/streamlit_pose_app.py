import os
# 设置环境变量解决OpenMP多重初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from webcam_rtmw_demo import Config, process_one_image, init_detector, init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.utils import adapt_mmdet_pipeline
import mmcv
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import math
import tempfile
matplotlib.use('Agg')  # 非交互式后端

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun', 'sans-serif']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体族

# 尝试使用系统中文字体
try:
    import platform
    
    # 根据操作系统设置适当的中文字体
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows下的黑体字体
    elif platform.system() == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/PingFang.ttc'
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    
    # 检查字体文件是否存在
    if os.path.exists(font_path):
        # 创建字体属性对象
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [chinese_font.get_name()]
        print(f"已加载中文字体: {chinese_font.get_name()}")
except Exception as e:
    print(f"加载系统中文字体失败: {str(e)}，使用默认字体设置")

# 设置页面配置
st.set_page_config(
    page_title="实时体态分析系统",
    page_icon="🏃‍♂️",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
    .metric-normal {
        color: #28a745;
    }
    .metric-warning {
        color: #dc3545;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.3rem;
    }
    .metric-value {
        margin: 0;
        font-size: 0.9rem;
    }
    .metric-title {
        margin: 0 0 0.2rem 0;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .metrics-container {
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .metrics-group-title {
        margin: 0.5rem 0 0.2rem 0;
        padding: 0.2rem 0.5rem;
        background-color: #e6f0ff;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
        grid-column: 1 / -1;
    }
    .metrics-summary {
        background-color: #f8f9fa;
        border-left: 3px solid #1f77b4;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .value-badge {
        display: inline-block;
        padding: 0.1rem 0.3rem;
        border-radius: 0.2rem;
        margin-right: 0.2rem;
        font-weight: bold;
    }
    .normal-badge {
        background-color: rgba(40, 167, 69, 0.2);
    }
    .warning-badge {
        background-color: rgba(220, 53, 69, 0.2);
    }
    .gait-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .gait-metric-card {
        background-color: #f8f9fa;
        border-radius: 0.3rem;
        padding: 0.5rem;
        text-align: center;
        border-top: 3px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    .gait-metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 0.2rem;
    }
    .gait-metric-title {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .gait-summary {
        background-color: #f0f7ff;
        border-left: 3px solid #1f77b4;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        border-radius: 0.3rem;
    }
    .symmetry-indicator {
        display: inline-block;
        width: 100%;
        height: 0.5rem;
        background: linear-gradient(to right, #ff7f0e, #1f77b4);
        border-radius: 1rem;
        margin: 0.3rem 0;
        position: relative;
    }
    .symmetry-marker {
        position: absolute;
        width: 0.6rem;
        height: 0.6rem;
        background-color: #333;
        border-radius: 50%;
        top: -0.05rem;
        transform: translateX(-50%);
    }
    .symmetry-label {
        font-size: 0.7rem;
        color: #666;
        display: flex;
        justify-content: space-between;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def check_value_in_range(value, range_str):
    """检查值是否在正常范围内"""
    if value is None:
        return False
    
    if '～' in range_str:
        parts = range_str.replace('°', '').split('～')
        min_val = float(parts[0].replace('%', ''))
        max_val = float(parts[1].replace('%', ''))
        return min_val <= value <= max_val
    elif '<' in range_str:
        max_val = float(range_str.replace('<', '').replace('°', '').replace('%', ''))
        return value < max_val
    elif '>' in range_str:
        min_val = float(range_str.replace('>', '').replace('°', '').replace('%', ''))
        return value > min_val
    return False

def analyze_gait_metrics(keypoints, keypoint_scores, prev_frame_data=None):
    """分析步态相关指标
    
    Args:
        keypoints: 关键点坐标数组，形状为(N, 2)
        keypoint_scores: 关键点置信度，形状为(N,)
        prev_frame_data: 前一帧的数据，用于计算时间差异
    
    Returns:
        dict: 包含步态测量值的字典
    """
    # 关键点索引 (根据RTMPose全身模型)
    LEFT_ANKLE_IDX = 15
    RIGHT_ANKLE_IDX = 16
    LEFT_KNEE_IDX = 13
    RIGHT_KNEE_IDX = 14
    LEFT_HIP_IDX = 11
    RIGHT_HIP_IDX = 12
    
    # 从prev_frame_data获取已经累积的指标值，如果没有则初始化为0
    results = {
        '左腿抬起时间': prev_frame_data.get('左腿抬起时间', 0),
        '右腿抬起时间': prev_frame_data.get('右腿抬起时间', 0),
        '双支撑时间': prev_frame_data.get('双支撑时间', 0),
        '步时': prev_frame_data.get('步时', 0),
        '摆动时间': prev_frame_data.get('摆动时间', 0),
        '支撑时间': prev_frame_data.get('支撑时间', 0)
    }

    # 定义关键点有效性检查函数
    def is_valid(idx):
        return idx < len(keypoint_scores) and keypoint_scores[idx] > 0.3  # 降低阈值，提高检测概率
    
    # 没有前一帧数据时返回空结果
    if prev_frame_data is None:
        return results
    
    # 检查所需关键点是否有效
    if not all(is_valid(idx) for idx in [LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, LEFT_KNEE_IDX, RIGHT_KNEE_IDX]):
        return results
    
    # 获取当前帧的时间戳
    current_time = time.time()
    
    # 计算脚踝和膝盖高度，用于判断腿是否抬起
    left_ankle_y = keypoints[LEFT_ANKLE_IDX][1]
    right_ankle_y = keypoints[RIGHT_ANKLE_IDX][1]
    left_knee_y = keypoints[LEFT_KNEE_IDX][1]
    right_knee_y = keypoints[RIGHT_KNEE_IDX][1]
    
    # 获取前一帧的数据
    prev_left_ankle_y = prev_frame_data.get('left_ankle_y')
    prev_right_ankle_y = prev_frame_data.get('right_ankle_y')
    prev_left_knee_y = prev_frame_data.get('left_knee_y', 0)
    prev_right_knee_y = prev_frame_data.get('right_knee_y', 0)
    prev_time = prev_frame_data.get('timestamp')
    
    # 计算时间差
    if prev_time:
        time_diff = current_time - prev_time
        
        # 新的判断逻辑：比较左右脚踝的高度差
        # 在图像坐标系中，Y轴向下为正，向上为负，所以Y值越小表示位置越高
        # 左脚踝比右脚踝高（左脚踝Y值小于右脚踝Y值），表示左腿抬起
        # 右脚踝比左脚踝高（右脚踝Y值小于左脚踝Y值），表示右腿抬起
        left_leg_up = left_ankle_y < right_ankle_y
        right_leg_up = right_ankle_y < left_ankle_y
        
        # 将腿的状态存储到prev_frame_data中
        prev_frame_data['left_leg_up'] = left_leg_up
        prev_frame_data['right_leg_up'] = right_leg_up
        
        # 存储当前脚踝高度差的信息
        prev_frame_data['ankle_height_diff'] = right_ankle_y - left_ankle_y
        
        # 其它数据仍需保存用于调试
        prev_frame_data['left_knee_ankle_distance'] = abs(left_knee_y - left_ankle_y)
        prev_frame_data['right_knee_ankle_distance'] = abs(right_knee_y - right_ankle_y)
        prev_frame_data['prev_left_knee_ankle_distance'] = abs(prev_left_knee_y - prev_left_ankle_y) if prev_left_knee_y is not None and prev_left_ankle_y is not None else abs(left_knee_y - left_ankle_y)
        prev_frame_data['prev_right_knee_ankle_distance'] = abs(prev_right_knee_y - prev_right_ankle_y) if prev_right_knee_y is not None and prev_right_ankle_y is not None else abs(right_knee_y - right_ankle_y)
        
        # 判断双腿是否都在地面上
        both_legs_down = not left_leg_up and not right_leg_up
        
        # 更新步态指标，将时间差累加到已有的指标上
        if left_leg_up:
            results['左腿抬起时间'] += time_diff
            results['摆动时间'] += time_diff
        
        if right_leg_up:
            results['右腿抬起时间'] += time_diff
            results['摆动时间'] += time_diff
        
        if both_legs_down:
            results['双支撑时间'] += time_diff
            results['支撑时间'] += time_diff
        
        # 步时是完整步态周期的时间
        results['步时'] += time_diff
    
    # 添加调试信息以帮助确认关键点的坐标是否被正确检测和更新
    print(f"左脚踝Y坐标: {left_ankle_y}, 右脚踝Y坐标: {right_ankle_y}")
    print(f"左膝盖Y坐标: {left_knee_y}, 右膝盖Y坐标: {right_knee_y}")
    print(f"脚踝高度差: {prev_frame_data['ankle_height_diff'] if 'ankle_height_diff' in prev_frame_data else '未知'}")
    
    # 添加更多调试信息，确认腿部抬起状态是否正确
    print(f"左腿抬起状态: {left_leg_up}, 右腿抬起状态: {right_leg_up}")
    print(f"左腿抬起时间: {results['左腿抬起时间']}, 右腿抬起时间: {results['右腿抬起时间']}")
    
    # 确保前一帧数据正确保存
    prev_frame_data['prev_left_ankle_y'] = left_ankle_y
    prev_frame_data['prev_right_ankle_y'] = right_ankle_y
    prev_frame_data['prev_left_knee_y'] = left_knee_y
    prev_frame_data['prev_right_knee_y'] = right_knee_y
    
    return results

def calculate_gait_symmetry(left_time, right_time):
    """计算步态对称性百分比
    
    Args:
        left_time: 左腿时间
        right_time: 右腿时间
    
    Returns:
        float: 对称性百分比 (0-100)，50%表示完全对称
    """
    if left_time <= 0 or right_time <= 0:
        return 50  # 默认对称
    
    total = left_time + right_time
    left_percentage = (left_time / total) * 100
    
    # 50%表示完美对称
    # 返回一个0-100的值，其中50表示完全对称
    # 0表示完全右偏，100表示完全左偏
    return left_percentage

def generate_gait_summary(gait_metrics, gait_normal_ranges):
    """生成步态分析摘要
    
    Args:
        gait_metrics: 步态指标数据
        gait_normal_ranges: 正常范围参考值
    
    Returns:
        str: 步态分析摘要HTML
    """
    # 计算对称性
    symmetry = calculate_gait_symmetry(
        gait_metrics['左腿抬起时间'], 
        gait_metrics['右腿抬起时间']
    )
    
    # 确定对称性的描述
    symmetry_percentage = abs(symmetry - 50) * 2  # 0-100%，0%是完全对称
    if symmetry_percentage < 10:
        symmetry_text = "步态对称性良好"
        symmetry_advice = "继续保持当前步态节奏"
    elif symmetry_percentage < 20:
        symmetry_text = "步态轻微不对称"
        symmetry_advice = "建议注意平衡左右腿的运动"
    else:
        symmetry_text = "步态明显不对称"
        left_or_right = "左" if symmetry > 50 else "右"
        symmetry_advice = f"存在{left_or_right}侧偏好，建议调整步态平衡"
    
    # 生成对称性指示器的位置 (0-100%)
    marker_position = symmetry
    
    # 检查步时是否在正常范围内
    step_time = gait_metrics['步时']
    min_step, max_step = gait_normal_ranges['步时']
    
    if step_time < min_step:
        pace_text = "步频偏快"
        pace_advice = "可以适当放慢步频，增加稳定性"
    elif step_time > max_step:
        pace_text = "步频偏慢"
        pace_advice = "可以适当增加步频，提高活动效率"
    else:
        pace_text = "步频正常"
        pace_advice = "当前步频处于健康范围"
    
    # 生成总结HTML
    summary_html = f"""
    <div class="gait-summary">
        <strong>步态分析总结：</strong><br>
        1. {symmetry_text}：{symmetry_advice}<br>
        <div class="symmetry-label">
            <span>右腿偏好</span>
            <span>平衡</span>
            <span>左腿偏好</span>
        </div>
        <div class="symmetry-indicator">
            <div class="symmetry-marker" style="left: {marker_position}%;"></div>
        </div>
        <br>
        2. {pace_text}：{pace_advice}<br>
        步时: {round(step_time, 2)}秒 (正常范围: {min_step}-{max_step}秒)
    </div>
    """
    
    return summary_html

def display_gait_metric(title, value, unit="秒"):
    """显示单个步态指标的卡片"""
    value_rounded = round(value, 2) if value is not None else 0
    return f"""<div class="gait-metric-card">
    <div class="gait-metric-title">{title}</div>
    <div class="gait-metric-value">{value_rounded} {unit}</div>
</div>"""

def display_debug_variables(prev_frame_data):
    """显示步态分析中的关键变量和中间值，用于调试和分析"""
    
    with st.expander("关键点坐标 (像素值)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 左侧关键点")
            if prev_frame_data.get('left_ankle_y') is not None:
                st.markdown(f"左脚踝Y坐标: {round(prev_frame_data['left_ankle_y'], 2)}")
                if prev_frame_data.get('left_ankle_x') is not None:
                    st.markdown(f"左脚踝X坐标: {round(prev_frame_data['left_ankle_x'], 2)}")
            if prev_frame_data.get('left_knee_y') is not None:
                st.markdown(f"左膝盖Y坐标: {round(prev_frame_data['left_knee_y'], 2)}")
                if prev_frame_data.get('left_knee_x') is not None:
                    st.markdown(f"左膝盖X坐标: {round(prev_frame_data['left_knee_x'], 2)}")
            if prev_frame_data.get('left_hip_y') is not None:
                st.markdown(f"左髋部Y坐标: {round(prev_frame_data['left_hip_y'], 2)}")
                if prev_frame_data.get('left_hip_x') is not None:
                    st.markdown(f"左髋部X坐标: {round(prev_frame_data['left_hip_x'], 2)}")
                
        with col2:
            st.markdown("#### 右侧关键点")
            if prev_frame_data.get('right_ankle_y') is not None:
                st.markdown(f"右脚踝Y坐标: {round(prev_frame_data['right_ankle_y'], 2)}")
                if prev_frame_data.get('right_ankle_x') is not None:
                    st.markdown(f"右脚踝X坐标: {round(prev_frame_data['right_ankle_x'], 2)}")
            if prev_frame_data.get('right_knee_y') is not None:
                st.markdown(f"右膝盖Y坐标: {round(prev_frame_data['right_knee_y'], 2)}")
                if prev_frame_data.get('right_knee_x') is not None:
                    st.markdown(f"右膝盖X坐标: {round(prev_frame_data['right_knee_x'], 2)}")
            if prev_frame_data.get('right_hip_y') is not None:
                st.markdown(f"右髋部Y坐标: {round(prev_frame_data['right_hip_y'], 2)}")
                if prev_frame_data.get('right_hip_x') is not None:
                    st.markdown(f"右髋部X坐标: {round(prev_frame_data['right_hip_x'], 2)}")
    
    with st.expander("步态检测值", expanded=False):
        st.markdown("#### 步态分析临时变量")
        
        # 显示脚踝高度差信息
        if prev_frame_data.get('ankle_height_diff') is not None:
            ankle_diff = prev_frame_data['ankle_height_diff']
            st.markdown(f"脚踝高度差 (右-左): {round(ankle_diff, 2)} 像素")
            st.markdown(f"左脚比右脚高: {'是' if ankle_diff > 0 else '否'}")
            st.markdown(f"右脚比左脚高: {'是' if ankle_diff < 0 else '否'}")
        
        # 显示腿部抬起状态
        if 'left_leg_up' in prev_frame_data:
            st.markdown(f"左腿抬起: {'是' if prev_frame_data['left_leg_up'] else '否'}")
        if 'right_leg_up' in prev_frame_data:
            st.markdown(f"右腿抬起: {'是' if prev_frame_data['right_leg_up'] else '否'}")
        
        # 膝踝距离
        if 'left_knee_ankle_distance' in prev_frame_data:
            st.markdown(f"左膝踝距离: {round(prev_frame_data['left_knee_ankle_distance'], 2)} 像素")
        if 'right_knee_ankle_distance' in prev_frame_data:
            st.markdown(f"右膝踝距离: {round(prev_frame_data['right_knee_ankle_distance'], 2)} 像素")
        
        # 上一帧膝踝距离
        if 'prev_left_knee_ankle_distance' in prev_frame_data:
            st.markdown(f"前帧左膝踝距离: {round(prev_frame_data['prev_left_knee_ankle_distance'], 2)} 像素")
        if 'prev_right_knee_ankle_distance' in prev_frame_data:
            st.markdown(f"前帧右膝踝距离: {round(prev_frame_data['prev_right_knee_ankle_distance'], 2)} 像素")

def display_metric(title, value, normal_range, key):
    """显示单个指标的卡片，更加紧凑的版本"""
    if value is not None:
        unit = '%' if '肥胖度' in title else '°'
        is_normal = check_value_in_range(value, normal_range)
        status_class = "metric-normal" if is_normal else "metric-warning"
        badge_class = "normal-badge" if is_normal else "warning-badge"
        status_text = "正常" if is_normal else "异常"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 class="metric-title">{title}</h4>
            <p class="metric-value">
                <span class="value-badge {badge_class}">{value}{unit}</span>
                <span class="{status_class}">({status_text})</span> | 范围: {normal_range}
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("实时体态分析系统")
    
    # 初始化配置
    args = Config()
    
    # 侧边栏配置
    st.sidebar.title("配置选项")
    
    # 输入源选择（新增部分）
    st.sidebar.markdown("### 输入源")
    input_source = st.sidebar.radio("选择输入源", ["实时摄像头", "上传视频"], index=0)
    
    # 模型配置部分
    st.sidebar.markdown("### 模型配置")
    with st.sidebar.expander("检测器配置", expanded=False):
        args.det_cat_id = st.number_input("检测类别ID", value=0, min_value=0)
        args.bbox_thr = st.slider("边界框阈值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        args.nms_thr = st.slider("NMS阈值", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    
    with st.sidebar.expander("姿态估计配置", expanded=False):
        args.kpt_thr = st.slider("关键点阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # 可视化配置部分
    st.sidebar.markdown("### 可视化配置")
    args.draw_bbox = st.sidebar.checkbox("显示边界框", value=False)
    args.draw_keypoints = st.sidebar.checkbox("显示关键点", value=True)
    args.show_posture_analysis = st.sidebar.checkbox("显示姿态分析", value=True)
    
    with st.sidebar.expander("可视化参数", expanded=False):
        args.radius = st.slider("关键点半径", min_value=1, max_value=10, value=5)
        args.thickness = st.slider("线条粗细", min_value=1, max_value=10, value=3)
        args.alpha = st.slider("透明度", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
        args.draw_heatmap = st.checkbox("显示热图", value=False)
        args.show_kpt_idx = st.checkbox("显示关键点索引", value=False)
    
    # 自定义关键点配置
    with st.sidebar.expander("自定义关键点", expanded=False):
        args.draw_iliac_midpoint = st.checkbox("显示髂骨中点", value=True)
        args.draw_neck_midpoint = st.checkbox("显示颈椎中点", value=True)
        args.custom_keypoint_radius = st.slider("自定义关键点半径", min_value=1, max_value=10, value=6)
        args.custom_keypoint_thickness = st.slider("自定义连接线粗细", min_value=1, max_value=10, value=4)
    
    # 性能配置部分
    st.sidebar.markdown("### 性能配置")
    args.fps = st.sidebar.checkbox("显示FPS", value=True)
    args.device = st.sidebar.selectbox("运行设备", options=['cuda:0', 'cpu'], index=0)
    
    # 输出配置部分
    st.sidebar.markdown("### 输出配置")
    save_output = st.sidebar.checkbox("保存输出", value=False)
    if save_output:
        args.output_root = st.sidebar.text_input("输出目录", value="output")
        args.save_predictions = st.sidebar.checkbox("保存预测结果", value=False)
    else:
        args.output_root = ''
        args.save_predictions = False
    
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
    
    # 初始化步态分析数据
    if 'gait_history' not in st.session_state:
        st.session_state.gait_history = {
            '左腿抬起时间': [],
            '右腿抬起时间': [],
            '双支撑时间': [],
            '步时': [],
            '摆动时间': [],
            '支撑时间': [],
            '左脚踝高度': [],
            '右脚踝高度': []
        }
        
    # 步态分析的正常范围 (x, y) 形式表示最小值和最大值
    gait_normal_ranges = {
        '左腿抬起时间': (0.25, 0.6),    # 正常范围：250-600毫秒
        '右腿抬起时间': (0.25, 0.6),    # 正常范围：250-600毫秒
        '双支撑时间': (0.05, 0.25),     # 正常范围：50-250毫秒
        '步时': (0.8, 1.2),            # 正常范围：800-1200毫秒
        '摆动时间': (0.25, 0.55),       # 正常范围：250-550毫秒
        '支撑时间': (0.5, 0.8)         # 正常范围：500-800毫秒
    }
    
    # 更新频率控制，每10帧更新一次图表
    frame_counter = 0
    
    # 定义关键点索引
    LEFT_ANKLE_IDX = 15
    RIGHT_ANKLE_IDX = 16
    LEFT_KNEE_IDX = 13
    RIGHT_KNEE_IDX = 14
    LEFT_HIP_IDX = 11
    RIGHT_HIP_IDX = 12
    
    # 前一帧数据存储
    prev_frame_data = {
        'left_ankle_y': None,
        'right_ankle_y': None,
        'left_knee_y': None,
        'right_knee_y': None,
        'left_hip_y': None,
        'right_hip_y': None,
        'left_ankle_x': None,
        'right_ankle_x': None,
        'left_knee_x': None,
        'right_knee_x': None,
        'left_hip_x': None,
        'right_hip_x': None,
        'timestamp': time.time()
    }
    
    # 最大历史数据点数量
    max_history_points = 100
    
    # 初始化模型
    @st.cache_resource
    def load_models():
        """加载所需的模型"""
        detector = init_detector(
            args.det_config, args.det_checkpoint, device=args.device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        
        pose_estimator = init_pose_estimator(
            args.pose_config,
            args.pose_checkpoint,
            device=args.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))
        
        pose_estimator.cfg.visualizer.radius = args.radius
        pose_estimator.cfg.visualizer.alpha = args.alpha
        pose_estimator.cfg.visualizer.line_width = args.thickness
        
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
        
        return detector, pose_estimator, visualizer
    
    try:
        with st.spinner('正在加载模型...'):
            detector, pose_estimator, visualizer = load_models()
        
        # 初始化摄像头或处理上传的视频
        if input_source == "实时摄像头":
            # 创建三列布局，用于并排显示视频、体态分析报告和步态分析
            col1, col2, col3 = st.columns([1.2, 0.9, 0.9])
            
            # 第一列：视频显示
            with col1:
                st.subheader("实时视频分析")
                video_placeholder = st.empty()
            
            # 第二列：体态分析结果
            with col2:
                st.subheader("体态分析报告")
                metrics_placeholder = st.empty()
            
            # 第三列：步态分析图表和指标
            with col3:
                st.subheader("步态分析")
                gait_metrics_placeholder = st.empty()
                gait_chart_placeholder = st.empty()
            
            # 初始化摄像头
            cap = None
            try:
                for camera_id in range(3):  # 尝试多个摄像头ID (0, 1, 2)
                    st.info(f"尝试连接摄像头 ID: {camera_id}")
                    cap = cv2.VideoCapture(camera_id)
                    # 检查摄像头是否成功打开
                    if cap is not None and cap.isOpened():
                        st.success(f"成功连接摄像头 ID: {camera_id}")
                        break
                    else:
                        if cap is not None:
                            cap.release()
                        st.warning(f"无法连接摄像头 ID: {camera_id}")
                
                if cap is None or not cap.isOpened():
                    st.error("无法连接任何摄像头，请检查摄像头连接和权限设置")
                    st.stop()
            except Exception as e:
                st.error(f"连接摄像头时出错: {str(e)}")
                st.stop()
            
            # 视频写入器
            video_writer = None
            if args.output_root:
                mmengine.mkdir_or_exist(args.output_root)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    f"{args.output_root}/output.mp4",
                    fourcc,
                    25,  # FPS
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            # FPS计算变量
            fps_value = 0
            frame_count = 0
            start_time = time.time()
            
            while True:
                success, frame = cap.read()
                if not success:
                    st.error("无法读取摄像头画面")
                    break
                
                # 处理帧并获取预测结果
                pred_instances = process_one_image(
                    args, frame, detector, pose_estimator, visualizer, 0.001)
                
                # 获取可视化后的帧
                frame_vis = visualizer.get_image()
                
                # 计算并显示FPS
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧更新一次FPS
                    end_time = time.time()
                    fps_value = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                if args.fps:
                    cv2.putText(frame_vis, f"FPS: {fps_value:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                
                # 显示处理后的帧
                video_placeholder.image(frame_vis, channels="RGB", use_container_width=True)
                
                # 保存视频
                if video_writer is not None:
                    video_writer.write(cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))
                
                # 分析体态并显示结果
                if hasattr(pred_instances, 'pred_instances') and len(pred_instances.pred_instances) > 0:
                    keypoints = pred_instances.pred_instances.keypoints[0]
                    keypoint_scores = pred_instances.pred_instances.keypoint_scores[0]
                    
                    custom_kpts = None
                    if hasattr(pred_instances, 'custom_keypoints') and len(pred_instances.custom_keypoints) > 0:
                        custom_kpts = pred_instances.custom_keypoints[0]
                    
                    # 分析体态
                    from webcam_rtmw_demo import analyze_body_posture
                    posture_results = analyze_body_posture(keypoints, keypoint_scores, custom_kpts)
                    
                    # 更新指标显示
                    with metrics_placeholder.container():
                        # 计算异常指标数量
                        abnormal_metrics = []
                        for title, value in posture_results.items():
                            if value is not None and not check_value_in_range(value, normal_ranges[title]):
                                abnormal_metrics.append(title)
                        
                        # 显示总结
                        total_metrics = sum(1 for v in posture_results.values() if v is not None)
                        if total_metrics > 0:
                            abnormal_count = len(abnormal_metrics)
                            normal_count = total_metrics - abnormal_count
                            if abnormal_count > 0:
                                advice = "建议关注以下异常指标并进行相应的调整和训练。"
                                abnormal_text = "、".join(abnormal_metrics[:3])
                                if len(abnormal_metrics) > 3:
                                    abnormal_text += f"等{len(abnormal_metrics)}项"
                            else:
                                advice = "您的体态状况良好，请继续保持。"
                                abnormal_text = ""
                            
                            summary = f"""<div class="metrics-summary">
                                检测到{total_metrics}项指标，其中{normal_count}项正常，{abnormal_count}项异常。{advice}
                                {f'<br><span class="metric-warning">异常项: {abnormal_text}</span>' if abnormal_count > 0 else ''}
                            </div>"""
                            st.markdown(summary, unsafe_allow_html=True)
                        
                        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
                        # 按分类组织指标
                        metrics_grouped = {
                            "头部": ["头前倾角", "头侧倾角", "头旋转角"],
                            "上半身": ["肩倾斜角", "圆肩角", "背部角"],
                            "中部": ["腹部肥胖度", "腰曲度", "骨盆前倾角", "侧中位度"],
                            "下肢": ["腿型-左腿", "腿型-右腿", "左膝评估角", "右膝评估角", "身体倾斜度", "足八角"]
                        }
                        
                        # 显示按组分类的指标
                        for group, metrics in metrics_grouped.items():
                            metrics_in_group = [m for m in metrics if m in posture_results and posture_results[m] is not None]
                            if metrics_in_group:
                                st.markdown(f'<div class="metrics-group-title">{group}</div>', unsafe_allow_html=True)
                                for metric in metrics_in_group:
                                    display_metric(metric, posture_results[metric], normal_ranges[metric], metric)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 分析步态数据
                    if LEFT_ANKLE_IDX < len(keypoints) and RIGHT_ANKLE_IDX < len(keypoints):
                        # 保存上一帧的数据
                        if prev_frame_data.get('left_ankle_y') is not None:
                            prev_frame_data['prev_left_ankle_y'] = prev_frame_data['left_ankle_y']
                            prev_frame_data['prev_left_ankle_x'] = prev_frame_data.get('left_ankle_x')
                        if prev_frame_data.get('right_ankle_y') is not None:
                            prev_frame_data['prev_right_ankle_y'] = prev_frame_data['right_ankle_y']
                            prev_frame_data['prev_right_ankle_x'] = prev_frame_data.get('right_ankle_x')
                        if prev_frame_data.get('left_knee_y') is not None:
                            prev_frame_data['prev_left_knee_y'] = prev_frame_data['left_knee_y']
                            prev_frame_data['prev_left_knee_x'] = prev_frame_data.get('left_knee_x')
                        if prev_frame_data.get('right_knee_y') is not None:
                            prev_frame_data['prev_right_knee_y'] = prev_frame_data['right_knee_y']
                            prev_frame_data['prev_right_knee_x'] = prev_frame_data.get('right_knee_x')
                        if prev_frame_data.get('left_hip_y') is not None:
                            prev_frame_data['prev_left_hip_y'] = prev_frame_data['left_hip_y']
                            prev_frame_data['prev_left_hip_x'] = prev_frame_data.get('left_hip_x')
                        if prev_frame_data.get('right_hip_y') is not None:
                            prev_frame_data['prev_right_hip_y'] = prev_frame_data['right_hip_y']
                            prev_frame_data['prev_right_hip_x'] = prev_frame_data.get('right_hip_x')
                        
                        # 更新当前帧数据
                        prev_frame_data['left_ankle_y'] = keypoints[LEFT_ANKLE_IDX][1]
                        prev_frame_data['left_ankle_x'] = keypoints[LEFT_ANKLE_IDX][0]
                        prev_frame_data['right_ankle_y'] = keypoints[RIGHT_ANKLE_IDX][1]
                        prev_frame_data['right_ankle_x'] = keypoints[RIGHT_ANKLE_IDX][0]
                        
                        # 添加膝盖坐标
                        if LEFT_KNEE_IDX < len(keypoints) and RIGHT_KNEE_IDX < len(keypoints):
                            prev_frame_data['left_knee_y'] = keypoints[LEFT_KNEE_IDX][1]
                            prev_frame_data['left_knee_x'] = keypoints[LEFT_KNEE_IDX][0]
                            prev_frame_data['right_knee_y'] = keypoints[RIGHT_KNEE_IDX][1]
                            prev_frame_data['right_knee_x'] = keypoints[RIGHT_KNEE_IDX][0]
                        
                        # 添加髋部坐标
                        if LEFT_HIP_IDX < len(keypoints) and RIGHT_HIP_IDX < len(keypoints):
                            prev_frame_data['left_hip_y'] = keypoints[LEFT_HIP_IDX][1]
                            prev_frame_data['left_hip_x'] = keypoints[LEFT_HIP_IDX][0]
                            prev_frame_data['right_hip_y'] = keypoints[RIGHT_HIP_IDX][1]
                            prev_frame_data['right_hip_x'] = keypoints[RIGHT_HIP_IDX][0]
                    
                    gait_metrics = analyze_gait_metrics(keypoints, keypoint_scores, prev_frame_data)
                    prev_frame_data.update(gait_metrics)
                    prev_frame_data['timestamp'] = time.time()
                    
                    # 显示当前步态指标
                    with gait_metrics_placeholder.container():
                        # 生成步态总结
                        summary_html = generate_gait_summary(gait_metrics, gait_normal_ranges)
                        st.markdown(summary_html, unsafe_allow_html=True)
                        
                        # 显示所有指标的卡片
                        col1, col2 = st.columns(2)
                        
                        # 左右腿指标放在第一列
                        with col1:
                            st.markdown("<h4 style='font-size:1rem;'>左右腿指标</h4>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                <div style='font-size:0.8rem;color:#666;'>左腿抬起时间</div>
                                <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['左腿抬起时间'], 2)} 秒</div>
                            </div>
                            <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                <div style='font-size:0.8rem;color:#666;'>右腿抬起时间</div>
                                <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['右腿抬起时间'], 2)} 秒</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 步态时间指标放在第二列
                        with col2:
                            st.markdown("<h4 style='font-size:1rem;'>步态时间指标</h4>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                <div style='font-size:0.8rem;color:#666;'>双支撑时间</div>
                                <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['双支撑时间'], 2)} 秒</div>
                            </div>
                            <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                <div style='font-size:0.8rem;color:#666;'>步时</div>
                                <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['步时'], 2)} 秒</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 其他时间指标
                        st.markdown("<h4 style='font-size:1rem;'>周期指标</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style='display:flex;gap:0.5rem;'>
                            <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;flex:1;'>
                                <div style='font-size:0.8rem;color:#666;'>摆动时间</div>
                                <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['摆动时间'], 2)} 秒</div>
                            </div>
                            <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;flex:1;'>
                                <div style='font-size:0.8rem;color:#666;'>支撑时间</div>
                                <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['支撑时间'], 2)} 秒</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 显示步态关键变量的值（新增部分）
                        display_debug_variables(prev_frame_data)
                    
                    # 更新步态历史数据
                    frame_counter += 1
                    if frame_counter % 1 == 0:  # 每10帧更新一次
                        for metric, value in gait_metrics.items():
                            st.session_state.gait_history[metric].append(value)
                            # 限制历史数据量
                            if len(st.session_state.gait_history[metric]) > max_history_points:
                                st.session_state.gait_history[metric] = st.session_state.gait_history[metric][-max_history_points:]
                        
                        # 添加脚踝高度数据
                        if 'left_ankle_y' in prev_frame_data and prev_frame_data['left_ankle_y'] is not None:
                            st.session_state.gait_history['左脚踝高度'].append(prev_frame_data['left_ankle_y'])
                            if len(st.session_state.gait_history['左脚踝高度']) > max_history_points:
                                st.session_state.gait_history['左脚踝高度'] = st.session_state.gait_history['左脚踝高度'][-max_history_points:]
                        
                        if 'right_ankle_y' in prev_frame_data and prev_frame_data['right_ankle_y'] is not None:
                            st.session_state.gait_history['右脚踝高度'].append(prev_frame_data['right_ankle_y'])
                            if len(st.session_state.gait_history['右脚踝高度']) > max_history_points:
                                st.session_state.gait_history['右脚踝高度'] = st.session_state.gait_history['右脚踝高度'][-max_history_points:]
                    
                        # 显示步态图表
                        with gait_chart_placeholder.container():
                            # 创建三行一列的图表布局
                            fig, axes = plt.subplots(3, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
                            
                            # 创建DataFrame用于绘图
                            df = pd.DataFrame(st.session_state.gait_history)
                            
                            # 为了可视化效果，计算移动平均
                            if len(df) > 5:  # 至少有5个数据点才能进行移动平均
                                df_smoothed = df.rolling(window=5, min_periods=1).mean()
                                
                                # 获取中文字体对象
                                try:
                                    font_prop = chinese_font
                                except NameError:
                                    font_prop = FontProperties(family=['SimHei', 'Microsoft YaHei'])
                                
                                # 第一个图表：左右腿的抬起时间对比
                                axes[0].plot(df_smoothed['左腿抬起时间'], label='左腿抬起时间', color='#1f77b4')
                                axes[0].plot(df_smoothed['右腿抬起时间'], label='右腿抬起时间', color='#ff7f0e', linestyle='--')
                                axes[0].set_title("左右腿抬起时间对比", fontproperties=font_prop, fontsize=12)
                                axes[0].set_ylabel("时间 (秒)", fontproperties=font_prop)
                                axes[0].legend(loc='upper left', fontsize='small', prop=font_prop)
                                axes[0].grid(True, linestyle='--', alpha=0.7)
                                
                                # 添加正常范围参考线
                                min_val, max_val = gait_normal_ranges['左腿抬起时间']
                                axes[0].axhspan(min_val, max_val, alpha=0.2, color='green', label='正常范围')
                                
                                # 第二个图表：步态周期相关时间
                                axes[1].plot(df_smoothed['双支撑时间'], label='双支撑时间', color='#2ca02c')
                                axes[1].plot(df_smoothed['步时'], label='步时', color='#d62728')
                                axes[1].plot(df_smoothed['摆动时间'], label='摆动时间', color='#9467bd')
                                axes[1].plot(df_smoothed['支撑时间'], label='支撑时间', color='#8c564b')
                                axes[1].set_title("步态周期时间分析", fontproperties=font_prop, fontsize=12)
                                axes[1].set_ylabel("时间 (秒)", fontproperties=font_prop)
                                axes[1].legend(loc='upper left', fontsize='small', prop=font_prop)
                                axes[1].grid(True, linestyle='--', alpha=0.7)
                                
                                # 第三个图表：左右脚踝高度
                                if '左脚踝高度' in df_smoothed and '右脚踝高度' in df_smoothed:
                                    # 将Y轴翻转，使得值越小（向上移动）显示在图表上方
                                    axes[2].plot(df_smoothed['左脚踝高度'], label='左脚踝高度', color='#17becf')
                                    axes[2].plot(df_smoothed['右脚踝高度'], label='右脚踝高度', color='#e377c2', linestyle='--')
                                    axes[2].set_title("左右脚踝高度变化", fontproperties=font_prop, fontsize=12)
                                    axes[2].set_xlabel("帧", fontproperties=font_prop)
                                    axes[2].set_ylabel("像素坐标 (Y轴)", fontproperties=font_prop)
                                    axes[2].legend(loc='upper left', fontsize='small', prop=font_prop)
                                    axes[2].grid(True, linestyle='--', alpha=0.7)
                                    # 因为在图像中Y坐标是向下增加的，所以翻转Y轴使得数值越小显示在上方
                                    axes[2].invert_yaxis()
                                
                                # 设置图表字体和样式
                                for ax in axes:
                                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                                        label.set_fontproperties(font_prop)
                                
                                # 应用紧凑布局
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("收集更多数据点以显示步态图表...")
            
            # 控制刷新率
            if input_source == "实时摄像头":
                time.sleep(0.1)
                
        elif input_source == "上传视频":
            # 创建上传视频区域
            st.markdown("### 上传视频")
            uploaded_video = st.file_uploader("选择视频文件", type=["mp4", "avi", "mov", "mkv"])
            
            if uploaded_video is not None:
                # 保存上传的视频到临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
                
                # 创建三列布局，用于并排显示视频、体态分析报告和步态分析
                col1, col2, col3 = st.columns([1.2, 0.9, 0.9])
                
                # 第一列：视频显示
                with col1:
                    st.subheader("视频分析")
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # 第二列：体态分析结果
                with col2:
                    st.subheader("体态分析报告")
                    metrics_placeholder = st.empty()
                
                # 第三列：步态分析图表和指标
                with col3:
                    st.subheader("步态分析")
                    gait_metrics_placeholder = st.empty()
                    gait_chart_placeholder = st.empty()
                
                # 处理视频按钮
                process_button = st.button("处理视频")
                
                if process_button:
                    # 处理视频并生成带有关键点的输出视频
                    try:
                        with st.spinner('正在加载模型...'):
                            detector, pose_estimator, visualizer = load_models()
                        
                        # 视频写入器
                        output_video_path = f"output_{int(time.time())}.mp4"
                        
                        # 打开视频文件
                        cap = cv2.VideoCapture(video_path)
                        
                        if not cap.isOpened():
                            st.error("无法打开视频文件")
                            st.stop()
                        
                        # 获取视频属性
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # 初始化视频写入器
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            output_video_path,
                            fourcc,
                            fps,
                            (width, height))
                        
                        # 初始化前一帧数据
                        prev_frame_data = {
                            'left_ankle_y': None,
                            'right_ankle_y': None,
                            'left_knee_y': None,
                            'right_knee_y': None,
                            'left_hip_y': None,
                            'right_hip_y': None,
                            'left_ankle_x': None,
                            'right_ankle_x': None,
                            'left_knee_x': None,
                            'right_knee_x': None,
                            'left_hip_x': None,
                            'right_hip_x': None,
                            'timestamp': time.time()
                        }
                        
                        # 存储每帧的分析结果
                        frame_results = []
                        
                        # 开始处理视频
                        frame_index = 0
                        while True:
                            success, frame = cap.read()
                            if not success:
                                break
                            
                            # 更新进度条和状态文本
                            progress = frame_index / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"处理进度: {int(progress * 100)}% (帧 {frame_index+1}/{total_frames})")
                            
                            # 处理帧并获取预测结果
                            pred_instances = process_one_image(
                                args, frame, detector, pose_estimator, visualizer, 0.001)
                            
                            # 获取可视化后的帧
                            frame_vis = visualizer.get_image()
                            
                            # 显示当前处理的帧
                            if frame_index % 5 == 0:  # 每5帧更新一次UI以减少卡顿
                                video_placeholder.image(frame_vis, channels="RGB", use_container_width=True)
                            
                            # 保存处理后的帧到输出视频
                            video_writer.write(cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))
                            
                            # 分析体态和步态
                            frame_result = {}
                            if hasattr(pred_instances, 'pred_instances') and len(pred_instances.pred_instances) > 0:
                                keypoints = pred_instances.pred_instances.keypoints[0]
                                keypoint_scores = pred_instances.pred_instances.keypoint_scores[0]
                                
                                custom_kpts = None
                                if hasattr(pred_instances, 'custom_keypoints') and len(pred_instances.custom_keypoints) > 0:
                                    custom_kpts = pred_instances.custom_keypoints[0]
                                
                                # 分析体态
                                from webcam_rtmw_demo import analyze_body_posture
                                posture_results = analyze_body_posture(keypoints, keypoint_scores, custom_kpts)
                                frame_result['posture'] = posture_results
                                
                                # 分析步态
                                if LEFT_ANKLE_IDX < len(keypoints) and RIGHT_ANKLE_IDX < len(keypoints):
                                    # 保存和更新关节点数据
                                    if prev_frame_data.get('left_ankle_y') is not None:
                                        prev_frame_data['prev_left_ankle_y'] = prev_frame_data['left_ankle_y']
                                        prev_frame_data['prev_left_ankle_x'] = prev_frame_data.get('left_ankle_x')
                                    if prev_frame_data.get('right_ankle_y') is not None:
                                        prev_frame_data['prev_right_ankle_y'] = prev_frame_data['right_ankle_y']
                                        prev_frame_data['prev_right_ankle_x'] = prev_frame_data.get('right_ankle_x')
                                    if prev_frame_data.get('left_knee_y') is not None:
                                        prev_frame_data['prev_left_knee_y'] = prev_frame_data['left_knee_y']
                                        prev_frame_data['prev_left_knee_x'] = prev_frame_data.get('left_knee_x')
                                    if prev_frame_data.get('right_knee_y') is not None:
                                        prev_frame_data['prev_right_knee_y'] = prev_frame_data['right_knee_y']
                                        prev_frame_data['prev_right_knee_x'] = prev_frame_data.get('right_knee_x')
                                    if prev_frame_data.get('left_hip_y') is not None:
                                        prev_frame_data['prev_left_hip_y'] = prev_frame_data['left_hip_y']
                                        prev_frame_data['prev_left_hip_x'] = prev_frame_data.get('left_hip_x')
                                    if prev_frame_data.get('right_hip_y') is not None:
                                        prev_frame_data['prev_right_hip_y'] = prev_frame_data['right_hip_y']
                                        prev_frame_data['prev_right_hip_x'] = prev_frame_data.get('right_hip_x')
                                    
                                    # 更新当前帧数据
                                    prev_frame_data['left_ankle_y'] = keypoints[LEFT_ANKLE_IDX][1]
                                    prev_frame_data['left_ankle_x'] = keypoints[LEFT_ANKLE_IDX][0]
                                    prev_frame_data['right_ankle_y'] = keypoints[RIGHT_ANKLE_IDX][1]
                                    prev_frame_data['right_ankle_x'] = keypoints[RIGHT_ANKLE_IDX][0]
                                    
                                    # 添加膝盖坐标
                                    if LEFT_KNEE_IDX < len(keypoints) and RIGHT_KNEE_IDX < len(keypoints):
                                        prev_frame_data['left_knee_y'] = keypoints[LEFT_KNEE_IDX][1]
                                        prev_frame_data['left_knee_x'] = keypoints[LEFT_KNEE_IDX][0]
                                        prev_frame_data['right_knee_y'] = keypoints[RIGHT_KNEE_IDX][1]
                                        prev_frame_data['right_knee_x'] = keypoints[RIGHT_KNEE_IDX][0]
                                    
                                    # 添加髋部坐标
                                    if LEFT_HIP_IDX < len(keypoints) and RIGHT_HIP_IDX < len(keypoints):
                                        prev_frame_data['left_hip_y'] = keypoints[LEFT_HIP_IDX][1]
                                        prev_frame_data['left_hip_x'] = keypoints[LEFT_HIP_IDX][0]
                                        prev_frame_data['right_hip_y'] = keypoints[RIGHT_HIP_IDX][1]
                                        prev_frame_data['right_hip_x'] = keypoints[RIGHT_HIP_IDX][0]
                                    
                                    gait_metrics = analyze_gait_metrics(keypoints, keypoint_scores, prev_frame_data)
                                    frame_result['gait'] = gait_metrics
                                    prev_frame_data.update(gait_metrics)
                            
                            # 保存帧结果
                            frame_results.append(frame_result)
                            
                            # 更新帧计数器
                            frame_index += 1
                        
                        # 完成视频处理
                        cap.release()
                        video_writer.release()
                        progress_bar.progress(1.0)
                        status_text.text("视频处理完成！")
                        
                        # 显示输出视频
                        st.subheader("处理后的视频")
                        st.video(output_video_path)
                        
                        # 提供下载链接
                        with open(output_video_path, 'rb') as file:
                            st.download_button(
                                label="下载处理后的视频",
                                data=file,
                                file_name=f"processed_video_{int(time.time())}.mp4",
                                mime="video/mp4"
                            )
                        
                        # 生成并显示分析报告
                        st.subheader("分析报告")
                        
                        # 创建选择特定帧的滑块
                        selected_frame = st.slider("选择帧", 0, len(frame_results)-1, 0)
                        
                        # 显示所选帧的分析结果
                        if frame_results and selected_frame < len(frame_results):
                            frame_result = frame_results[selected_frame]
                            
                            # 如果有体态数据，显示体态分析
                            if 'posture' in frame_result and frame_result['posture']:
                                st.subheader("体态分析")
                                posture_results = frame_result['posture']
                                
                                # 计算异常指标数量
                                abnormal_metrics = []
                                for title, value in posture_results.items():
                                    if value is not None and not check_value_in_range(value, normal_ranges[title]):
                                        abnormal_metrics.append(title)
                                
                                # 显示总结
                                total_metrics = sum(1 for v in posture_results.values() if v is not None)
                                if total_metrics > 0:
                                    abnormal_count = len(abnormal_metrics)
                                    normal_count = total_metrics - abnormal_count
                                    
                                    if abnormal_count > 0:
                                        advice = "建议关注以下异常指标并进行相应的调整和训练。"
                                        abnormal_text = "、".join(abnormal_metrics[:3])
                                        if len(abnormal_metrics) > 3:
                                            abnormal_text += f"等{len(abnormal_metrics)}项"
                                    else:
                                        advice = "您的体态状况良好，请继续保持。"
                                        abnormal_text = ""
                                    
                                    summary = f"""<div class="metrics-summary">
                                        检测到{total_metrics}项指标，其中{normal_count}项正常，{abnormal_count}项异常。{advice}
                                        {f'<br><span class="metric-warning">异常项: {abnormal_text}</span>' if abnormal_count > 0 else ''}
                                    </div>"""
                                    st.markdown(summary, unsafe_allow_html=True)
                                
                                # 按分类组织指标
                                metrics_grouped = {
                                    "头部": ["头前倾角", "头侧倾角", "头旋转角"],
                                    "上半身": ["肩倾斜角", "圆肩角", "背部角"],
                                    "中部": ["腹部肥胖度", "腰曲度", "骨盆前倾角", "侧中位度"],
                                    "下肢": ["腿型-左腿", "腿型-右腿", "左膝评估角", "右膝评估角", "身体倾斜度", "足八角"]
                                }
                                
                                # 显示按组分类的指标
                                for group, metrics in metrics_grouped.items():
                                    metrics_in_group = [m for m in metrics if m in posture_results and posture_results[m] is not None]
                                    if metrics_in_group:
                                        st.markdown(f'<div class="metrics-group-title">{group}</div>', unsafe_allow_html=True)
                                        for metric in metrics_in_group:
                                            display_metric(metric, posture_results[metric], normal_ranges[metric], metric)
                            
                            # 如果有步态数据，显示步态分析
                            if 'gait' in frame_result and frame_result['gait']:
                                st.subheader("步态分析")
                                gait_metrics = frame_result['gait']
                                
                                # 生成步态总结
                                summary_html = generate_gait_summary(gait_metrics, gait_normal_ranges)
                                st.markdown(summary_html, unsafe_allow_html=True)
                                
                                # 显示步态指标
                                col1, col2 = st.columns(2)
                                
                                # 左右腿指标
                                with col1:
                                    st.markdown("<h4 style='font-size:1rem;'>左右腿指标</h4>", unsafe_allow_html=True)
                                    st.markdown(f"""
                                    <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                        <div style='font-size:0.8rem;color:#666;'>左腿抬起时间</div>
                                        <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['左腿抬起时间'], 2)} 秒</div>
                                    </div>
                                    <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                        <div style='font-size:0.8rem;color:#666;'>右腿抬起时间</div>
                                        <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['右腿抬起时间'], 2)} 秒</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # 步态时间指标
                                with col2:
                                    st.markdown("<h4 style='font-size:1rem;'>步态时间指标</h4>", unsafe_allow_html=True)
                                    st.markdown(f"""
                                    <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                        <div style='font-size:0.8rem;color:#666;'>双支撑时间</div>
                                        <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['双支撑时间'], 2)} 秒</div>
                                    </div>
                                    <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;'>
                                        <div style='font-size:0.8rem;color:#666;'>步时</div>
                                        <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['步时'], 2)} 秒</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # 其他时间指标
                                st.markdown("<h4 style='font-size:1rem;'>周期指标</h4>", unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style='display:flex;gap:0.5rem;'>
                                    <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;flex:1;'>
                                        <div style='font-size:0.8rem;color:#666;'>摆动时间</div>
                                        <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['摆动时间'], 2)} 秒</div>
                                    </div>
                                    <div style='background-color:#f8f9fa;border-radius:0.3rem;padding:0.5rem;margin-bottom:0.5rem;border-top:3px solid #1f77b4;flex:1;'>
                                        <div style='font-size:0.8rem;color:#666;'>支撑时间</div>
                                        <div style='font-size:1.2rem;font-weight:bold;color:#1f77b4;'>{round(gait_metrics['支撑时间'], 2)} 秒</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 添加整体视频分析报告
                                if len(frame_results) > 5:  # 确保有足够的数据点
                                    st.subheader("整体视频分析趋势")
                                    
                                    # 提取所有帧的体态和步态数据
                                    posture_data = {}
                                    gait_data = {}
                                    
                                    # 初始化数据结构
                                    for i, frame_result in enumerate(frame_results):
                                        # 收集体态数据
                                        if 'posture' in frame_result and frame_result['posture']:
                                            for metric, value in frame_result['posture'].items():
                                                if value is not None:
                                                    if metric not in posture_data:
                                                        posture_data[metric] = []
                                                    # 填充可能缺失的前面数据点
                                                    while len(posture_data[metric]) < i:
                                                        posture_data[metric].append(None)
                                                    posture_data[metric].append(value)
                                        
                                        # 收集步态数据
                                        if 'gait' in frame_result and frame_result['gait']:
                                            for metric, value in frame_result['gait'].items():
                                                if metric not in gait_data:
                                                    gait_data[metric] = []
                                                # 填充可能缺失的前面数据点
                                                while len(gait_data[metric]) < i:
                                                    gait_data[metric].append(None)
                                                gait_data[metric].append(value)
                                    
                                    # 确保所有数据列表长度一致
                                    max_length = len(frame_results)
                                    for metric in posture_data:
                                        while len(posture_data[metric]) < max_length:
                                            posture_data[metric].append(None)
                                    
                                    for metric in gait_data:
                                        while len(gait_data[metric]) < max_length:
                                            gait_data[metric].append(None)
                                    
                                    # 创建数据帧
                                    posture_df = pd.DataFrame(posture_data)
                                    gait_df = pd.DataFrame(gait_data)
                                    
                                    # 绘制体态数据趋势图
                                    if not posture_df.empty and posture_df.shape[1] > 0:
                                        st.subheader("体态指标趋势")
                                        
                                        # 按组展示体态趋势图
                                        metrics_grouped = {
                                            "头部": ["头前倾角", "头侧倾角", "头旋转角"],
                                            "上半身": ["肩倾斜角", "圆肩角", "背部角"],
                                            "中部": ["腹部肥胖度", "腰曲度", "骨盆前倾角", "侧中位度"],
                                            "下肢": ["腿型-左腿", "腿型-右腿", "左膝评估角", "右膝评估角", "身体倾斜度", "足八角"]
                                        }
                                        
                                        for group_name, metrics in metrics_grouped.items():
                                            group_metrics = [m for m in metrics if m in posture_df.columns]
                                            if group_metrics:
                                                st.markdown(f"#### {group_name}指标趋势")
                                                
                                                # 计算每个指标的移动平均
                                                smoothed_df = posture_df[group_metrics].copy()
                                                for col in smoothed_df.columns:
                                                    smoothed_df[col] = smoothed_df[col].rolling(window=5, min_periods=1).mean()
                                                
                                                # 分批绘图，每批最多显示3个指标
                                                for i in range(0, len(group_metrics), 3):
                                                    batch_metrics = group_metrics[i:i+3]
                                                    if batch_metrics:
                                                        fig, ax = plt.subplots(figsize=(10, 5))
                                                        for metric in batch_metrics:
                                                            ax.plot(smoothed_df[metric], label=metric)
                                                        
                                                        # 添加正常范围区域
                                                        for metric in batch_metrics:
                                                            if metric in normal_ranges:
                                                                range_str = normal_ranges[metric]
                                                                min_val, max_val = None, None
                                                                
                                                                # 解析正常范围
                                                                if '～' in range_str:
                                                                    parts = range_str.split('～')
                                                                    min_part = parts[0].replace('°', '').replace('<', '').replace('>', '')
                                                                    max_part = parts[1].replace('°', '').replace('<', '').replace('>', '')
                                                                    
                                                                    try:
                                                                        min_val = float(min_part)
                                                                        max_val = float(max_part)
                                                                    except ValueError:
                                                                        pass
                                                                elif '<' in range_str:
                                                                    max_part = range_str.replace('°', '').replace('<', '')
                                                                    try:
                                                                        max_val = float(max_part)
                                                                        min_val = smoothed_df[metric].min() - 5  # 假设下限
                                                                    except ValueError:
                                                                        pass
                                                                elif '>' in range_str:
                                                                    min_part = range_str.replace('°', '').replace('>', '')
                                                                    try:
                                                                        min_val = float(min_part)
                                                                        max_val = smoothed_df[metric].max() + 5  # 假设上限
                                                                    except ValueError:
                                                                        pass
                                                                
                                                                if min_val is not None and max_val is not None:
                                                                    ax.axhspan(min_val, max_val, alpha=0.2, color='green', label=f"{metric}正常范围")
                                                
                                                ax.set_xlabel('帧')
                                                ax.set_ylabel('度数')
                                                ax.legend()
                                                ax.grid(True, linestyle='--', alpha=0.7)
                                                st.pyplot(fig)
                                
                                # 绘制步态数据趋势图
                                if not gait_df.empty and gait_df.shape[1] > 0:
                                    st.subheader("步态指标趋势")
                                    
                                    # 步态指标分组
                                    gait_groups = {
                                        "抬腿时间": ["左腿抬起时间", "右腿抬起时间"],
                                        "步态周期": ["双支撑时间", "步时"],
                                        "支撑周期": ["摆动时间", "支撑时间"]
                                    }
                                    
                                    for group_name, metrics in gait_groups.items():
                                        group_metrics = [m for m in metrics if m in gait_df.columns]
                                        if group_metrics:
                                            st.markdown(f"#### {group_name}趋势")
                                            
                                            # 计算移动平均
                                            smoothed_df = gait_df[group_metrics].copy()
                                            for col in smoothed_df.columns:
                                                smoothed_df[col] = smoothed_df[col].rolling(window=5, min_periods=1).mean()
                                            
                                            fig, ax = plt.subplots(figsize=(10, 5))
                                            for metric in group_metrics:
                                                ax.plot(smoothed_df[metric], label=metric)
                                            
                                            # 添加正常范围
                                            for metric in group_metrics:
                                                if metric in gait_normal_ranges:
                                                    min_val, max_val = gait_normal_ranges[metric]
                                                    ax.axhspan(min_val, max_val, alpha=0.2, color='green', label=f"{metric}正常范围")
                                            
                                            ax.set_xlabel('帧')
                                            ax.set_ylabel('时间 (秒)')
                                            ax.legend()
                                            ax.grid(True, linestyle='--', alpha=0.7)
                                            st.pyplot(fig)
                                
                                # 添加整体统计分析
                                st.subheader("整体统计分析")
                                
                                # 体态指标统计
                                if not posture_df.empty and posture_df.shape[1] > 0:
                                    st.markdown("#### 体态指标统计")
                                    
                                    # 为每个体态指标计算统计值
                                    stats_data = []
                                    for metric in posture_df.columns:
                                        if metric in normal_ranges:
                                            values = posture_df[metric].dropna()
                                            if len(values) > 0:
                                                mean_val = values.mean()
                                                abnormal_count = sum(1 for v in values if not check_value_in_range(v, normal_ranges[metric]))
                                                abnormal_pct = (abnormal_count / len(values)) * 100
                                                
                                                stats_data.append({
                                                    "指标": metric,
                                                    "平均值": f"{mean_val:.2f}°",
                                                    "正常范围": normal_ranges[metric],
                                                    "异常比例": f"{abnormal_pct:.1f}%"
                                                })
                                    
                                    if stats_data:
                                        stats_df = pd.DataFrame(stats_data)
                                        st.dataframe(stats_df)
                                
                                # 步态指标统计
                                if not gait_df.empty and gait_df.shape[1] > 0:
                                    st.markdown("#### 步态指标统计")
                                    
                                    # 为每个步态指标计算统计值
                                    stats_data = []
                                    for metric in gait_df.columns:
                                        if metric in gait_normal_ranges:
                                            values = gait_df[metric].dropna()
                                            if len(values) > 0:
                                                mean_val = values.mean()
                                                min_val, max_val = gait_normal_ranges[metric]
                                                abnormal_count = sum(1 for v in values if v < min_val or v > max_val)
                                                abnormal_pct = (abnormal_count / len(values)) * 100
                                                
                                                stats_data.append({
                                                    "指标": metric,
                                                    "平均值": f"{mean_val:.2f}秒",
                                                    "正常范围": f"{min_val}～{max_val}秒",
                                                    "异常比例": f"{abnormal_pct:.1f}%"
                                                })
                                    
                                    if stats_data:
                                        stats_df = pd.DataFrame(stats_data)
                                        st.dataframe(stats_df)
                                    
                                    # 步态对称性分析
                                    if "左腿抬起时间" in gait_df.columns and "右腿抬起时间" in gait_df.columns:
                                        left_times = gait_df["左腿抬起时间"].dropna()
                                        right_times = gait_df["右腿抬起时间"].dropna()
                                        
                                        if len(left_times) > 0 and len(right_times) > 0:
                                            left_mean = left_times.mean()
                                            right_mean = right_times.mean()
                                            symmetry = calculate_gait_symmetry(left_mean, right_mean)
                                            
                                            st.markdown("#### 步态对称性分析")
                                            symmetry_color = "green" if symmetry >= 90 else ("orange" if symmetry >= 80 else "red")
                                            st.markdown(f"""
                                            <div style='background-color:#f8f9fa;border-radius:0.5rem;padding:1rem;margin:1rem 0;border-left:5px solid {symmetry_color};'>
                                                <h4 style='margin-top:0;'>步态对称性指数</h4>
                                                <div style='font-size:2rem;font-weight:bold;color:{symmetry_color};'>{symmetry:.1f}%</div>
                                                <div style='font-size:0.9rem;color:#666;margin-top:0.5rem;'>
                                                    左腿抬起时间: {left_mean:.2f}秒 | 右腿抬起时间: {right_mean:.2f}秒
                                                </div>
                                                <div style='font-size:0.9rem;margin-top:0.5rem;'>
                                                    {
                                                        "步态对称性非常好，保持良好的步行姿势。" if symmetry >= 95 else
                                                        "步态对称性良好，轻微不平衡，可关注改善。" if symmetry >= 90 else
                                                        "步态存在中度不对称，建议进行针对性训练。" if symmetry >= 80 else
                                                        "步态严重不对称，建议咨询专业康复师。"
                                                    }
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"处理视频时出错: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # 清理临时文件
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.unlink(video_path)
                
        else:
            st.info("请上传视频文件进行分析")
            
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        st.stop()
    finally:
        if 'cap' in locals():
            cap.release()
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()

if __name__ == '__main__':
    main()