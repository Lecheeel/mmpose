import time

# 从config模块导入常量
from config import LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, LEFT_KNEE_IDX, RIGHT_KNEE_IDX

def analyze_gait_metrics(keypoints, keypoint_scores, prev_frame_data=None):
    """分析步态相关指标
    
    Args:
        keypoints: 关键点坐标数组，形状为(N, 2)
        keypoint_scores: 关键点置信度，形状为(N,)
        prev_frame_data: 前一帧的数据，用于计算时间差异
    
    Returns:
        dict: 包含步态测量值的字典
    """
    results = {
        '左腿抬起时间': 0,
        '右腿抬起时间': 0,
        '双支撑时间': 0,
        '步时': 0,
        '摆动时间': 0,
        '支撑时间': 0
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
        
        # 判断腿是否抬起 (使用多个条件组合判断)
        if prev_left_ankle_y is not None and prev_right_ankle_y is not None:
            # 降低阈值，检测更细微的变化
            threshold = 3  # 降低到3像素
            
            # 左腿抬起条件：脚踝Y坐标变小(向上移动)或者膝盖和脚踝之间的距离变小
            left_leg_up = (left_ankle_y < prev_left_ankle_y - threshold) or \
                          (abs(left_knee_y - left_ankle_y) < abs(prev_left_knee_y - prev_left_ankle_y) - threshold)
            
            # 右腿抬起条件：脚踝Y坐标变小(向上移动)或者膝盖和脚踝之间的距离变小
            right_leg_up = (right_ankle_y < prev_right_ankle_y - threshold) or \
                           (abs(right_knee_y - right_ankle_y) < abs(prev_right_knee_y - prev_right_ankle_y) - threshold)
            
            # 判断双腿是否都在地面上
            both_legs_down = not left_leg_up and not right_leg_up
            
            # 更新步态指标
            if left_leg_up:
                results['左腿抬起时间'] = prev_frame_data.get('左腿抬起时间', 0) + time_diff
                results['摆动时间'] = prev_frame_data.get('摆动时间', 0) + time_diff
            
            if right_leg_up:
                results['右腿抬起时间'] = prev_frame_data.get('右腿抬起时间', 0) + time_diff
                results['摆动时间'] = prev_frame_data.get('摆动时间', 0) + time_diff
            
            if both_legs_down:
                results['双支撑时间'] = prev_frame_data.get('双支撑时间', 0) + time_diff
                results['支撑时间'] = prev_frame_data.get('支撑时间', 0) + time_diff
            
            # 步时是完整步态周期的时间
            results['步时'] = prev_frame_data.get('步时', 0) + time_diff
    
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
    """显示单个步态指标的卡片
    
    Args:
        title: 指标标题
        value: 指标值
        unit: 单位，默认为"秒"
        
    Returns:
        str: HTML格式的指标卡片
    """
    value_rounded = round(value, 2) if value is not None else 0
    return f"""<div class="gait-metric-card">
    <div class="gait-metric-title">{title}</div>
    <div class="gait-metric-value">{value_rounded} {unit}</div>
</div>"""

def init_gait_history():
    """初始化步态历史数据
    
    Returns:
        dict: 包含空步态历史数据的字典
    """
    return {
        '左腿抬起时间': [],
        '右腿抬起时间': [],
        '双支撑时间': [],
        '步时': [],
        '摆动时间': [],
        '支撑时间': []
    }

def update_gait_history(gait_history, gait_metrics, max_history_points=100):
    """更新步态历史数据
    
    Args:
        gait_history: 现有的步态历史数据
        gait_metrics: 当前帧的步态指标
        max_history_points: 最大历史数据点数量，默认为100
        
    Returns:
        dict: 更新后的步态历史数据
    """
    for metric, value in gait_metrics.items():
        gait_history[metric].append(value)
        # 限制历史数据量
        if len(gait_history[metric]) > max_history_points:
            gait_history[metric] = gait_history[metric][-max_history_points:]
    
    return gait_history 