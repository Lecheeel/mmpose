import time
import cv2
import torch
import numpy as np
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.apis.inference_tracking import _compute_iou
from . import config
from .utils import torch_inference_mode, init_torch_settings
from .pose_visualization import process_pose_results

def camera_process(camera_id, return_dict, shared_data, model_name='rtmpose-l_8xb32-270e_coco-wholebody-384x288', device='cuda:0'):
    """每个摄像头的独立处理进程
    
    Args:
        camera_id: 摄像头ID
        return_dict: 用于返回处理结果的字典
        shared_data: 进程间共享数据
        model_name: 模型名称
        device: 设备名称
    """
    try:
        # 设置CUDNN加速
        init_torch_settings(device)
        
        # 初始化摄像头 - 使用DirectShow后端
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return_dict[f'error_{camera_id}'] = f"无法打开摄像头 {camera_id}"
            return
            
        # 尝试设置更高的捕获分辨率和其他优化
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 设置缓冲区大小为1，减少延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 设置合适的分辨率和帧率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DEFAULT_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DEFAULT_CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.DEFAULT_CAMERA_FPS)

        # 在独立进程中初始化模型
        try:
            inferencer = MMPoseInferencer(
                pose2d=model_name,
                device=device,
                scope='mmpose',
                show_progress=False
            )
            print(f"进程 {camera_id} 成功加载模型到 {device}")
            
            # 预热模型以初始化CUDA核心和缓存
            dummy_frame = np.zeros((config.DEFAULT_CAMERA_HEIGHT, config.DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8)
            with torch_inference_mode():
                for _ in range(10):  # 预热10次
                    _ = list(inferencer(dummy_frame))
                    
                # 强制同步GPU，确保预热完成    
                if 'cuda' in device:
                    torch.cuda.synchronize()
                
        except Exception as e:
            print(f"进程 {camera_id} 模型加载失败: {str(e)}")
            return_dict[f'error_{camera_id}'] = f"模型加载失败: {str(e)}"
            return
            
        # 推理配置 - 微调提高性能
        call_args = {
            'show': False,
            'draw_bbox': True,  
            'radius': 3,  # 减小关键点半径加快渲染
            'thickness': 1,  # 减小线条粗细加快渲染
            'kpt_thr': 0.3,  # 略微降低阈值提高检测能力
            'bbox_thr': 0.3,
            'nms_thr': 0.5,
            'pose_based_nms': True,
            'max_num_bboxes': 15
        }
        
        # 性能统计
        inference_times = []
        frame_count = 0
        start_time = time.time()
        
        # 创建图像转换缓存 - 预分配内存
        img_cache = np.empty((config.DEFAULT_CAMERA_HEIGHT, config.DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # 跟踪状态变量
        results_last = []  # 上一帧的结果
        next_id = 0        # 下一个可用的ID
        tracking_thr = config.TRACKING_THRESHOLD  # IOU阈值
        
        # 自定义跟踪函数
        def track_by_iou(bbox, results_last, thr):
            """使用IOU跟踪对象"""
            max_iou_score = -1
            max_index = -1
            track_id = -1
            
            # 确保bbox格式正确
            if not isinstance(bbox, (list, np.ndarray)) or len(bbox) < 4:
                return -1, results_last
            
            for index, res_last in enumerate(results_last):
                last_bbox = res_last.get('bbox', None)
                if last_bbox is None or not isinstance(last_bbox, (list, np.ndarray)) or len(last_bbox) < 4:
                    continue
                
                # 计算IOU，使用MMPose提供的函数
                try:
                    # 使用MMPose的IOU计算函数
                    iou_score = _compute_iou(bbox, last_bbox)
                    if iou_score > max_iou_score:
                        max_iou_score = iou_score
                        max_index = index
                except Exception as e:
                    print(f"计算IOU时出错: {str(e)}, bbox1: {bbox}, bbox2: {last_bbox}")
                    continue
            
            # 如果IOU得分大于阈值，使用匹配的上一帧对象的ID
            if max_iou_score > thr and max_index != -1:
                track_id = results_last[max_index].get('track_id', -1)
                # 从结果列表中移除已匹配的项
                results_last.pop(max_index)
            
            return track_id, results_last
        
        # 主处理循环
        while shared_data.running.value:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print(f"摄像头 {camera_id} 无法读取帧，尝试重新初始化...")
                # 尝试重新初始化摄像头
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print(f"摄像头 {camera_id} 无法重新打开，退出进程")
                    break
                # 重新设置摄像头参数
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DEFAULT_CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DEFAULT_CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.DEFAULT_CAMERA_FPS)
                continue
                
            frame_count += 1
            
            # 处理帧
            start_inference = time.time()
            
            # 如果帧尺寸不符合预期，重新调整图像缓存
            if img_cache.shape[:2] != frame.shape[:2]:
                img_cache = np.empty(frame.shape, dtype=np.uint8)
            
            # 直接在预分配的内存上进行颜色转换
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, img_cache)
            
            # 推理 - 使用优化的上下文管理器
            try:
                with torch_inference_mode():
                    raw_results = list(inferencer(img_cache, **call_args))
                    # 确保GPU操作完成，减少延迟抖动
                    if 'cuda' in device:
                        torch.cuda.synchronize()
            except Exception as e:
                print(f"推理过程中出错: {str(e)}")
                continue  # 跳过这一帧，继续处理下一帧
            
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            if len(inference_times) > 30:
                inference_times.pop(0)
            
            # 添加跟踪ID处理
            current_results = []
            
            try:
                for result in raw_results:
                    pred_instances = result.get('predictions', [])
                    
                    # 遍历pred_instances中的所有实例
                    if pred_instances and len(pred_instances) > 0:
                        # MMPose 2.x返回的结果结构可能是一个列表，包含多个模型的预测结果
                        # 我们需要确保正确处理这种结构
                        for instance_list in pred_instances:
                            # 检查instance_list是否为列表类型
                            if isinstance(instance_list, list):
                                # 遍历列表中的每个实例（每个人）
                                for instance in instance_list:
                                    # 获取边界框
                                    bbox = instance.get('bbox', None)
                                    keypoints = instance.get('keypoints', None)
                                    
                                    # 如果没有边界框但有关键点，则使用关键点创建一个边界框
                                    if bbox is None and keypoints is not None and len(keypoints) > 0:
                                        try:
                                            keypoints = np.array(keypoints)
                                            if keypoints.size > 0:
                                                # 计算包含所有关键点的边界框
                                                valid_mask = np.isfinite(keypoints).all(axis=1)
                                                if np.any(valid_mask):  # 确保至少有一个有效的关键点
                                                    valid_keypoints = keypoints[valid_mask]
                                                    x_min = np.min(valid_keypoints[:, 0])
                                                    y_min = np.min(valid_keypoints[:, 1])
                                                    x_max = np.max(valid_keypoints[:, 0])
                                                    y_max = np.max(valid_keypoints[:, 1])
                                                    bbox = [x_min, y_min, x_max, y_max]
                                                    instance['bbox'] = bbox
                                        except Exception as e:
                                            print(f"从关键点创建边界框时出错: {str(e)}")
                                    
                                    if bbox is not None:
                                        try:
                                            # 确保边界框格式正确
                                            if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                                # 尝试跟踪
                                                track_id, results_last = track_by_iou(bbox, results_last, tracking_thr)
                                                if track_id == -1:
                                                    # 如果没有匹配，分配新ID
                                                    track_id = next_id
                                                    next_id += 1
                                                
                                                # 设置跟踪ID
                                                instance['track_id'] = track_id
                                                
                                                # 保存当前实例以供下一帧使用
                                                current_results.append(instance)
                                        except Exception as e:
                                            print(f"处理跟踪ID时出错: {str(e)}, bbox: {bbox}")
                            elif isinstance(instance_list, dict):
                                # 如果直接是一个字典(单个人的情况)，直接处理
                                instance = instance_list
                                bbox = instance.get('bbox', None)
                                keypoints = instance.get('keypoints', None)
                                
                                # 如果没有边界框但有关键点，则使用关键点创建一个边界框
                                if bbox is None and keypoints is not None and len(keypoints) > 0:
                                    try:
                                        keypoints = np.array(keypoints)
                                        if keypoints.size > 0:
                                            # 计算包含所有关键点的边界框
                                            valid_mask = np.isfinite(keypoints).all(axis=1)
                                            if np.any(valid_mask):  # 确保至少有一个有效的关键点
                                                valid_keypoints = keypoints[valid_mask]
                                                x_min = np.min(valid_keypoints[:, 0])
                                                y_min = np.min(valid_keypoints[:, 1])
                                                x_max = np.max(valid_keypoints[:, 0])
                                                y_max = np.max(valid_keypoints[:, 1])
                                                bbox = [x_min, y_min, x_max, y_max]
                                                instance['bbox'] = bbox
                                    except Exception as e:
                                        print(f"从关键点创建边界框时出错: {str(e)}")
                                
                                if bbox is not None:
                                    try:
                                        # 确保边界框格式正确
                                        if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                            # 尝试跟踪
                                            track_id, results_last = track_by_iou(bbox, results_last, tracking_thr)
                                            if track_id == -1:
                                                # 如果没有匹配，分配新ID
                                                track_id = next_id
                                                next_id += 1
                                            
                                            # 设置跟踪ID
                                            instance['track_id'] = track_id
                                            
                                            # 保存当前实例以供下一帧使用
                                            current_results.append(instance)
                                    except Exception as e:
                                        print(f"处理跟踪ID时出错: {str(e)}, bbox: {bbox}")
            except Exception as e:
                print(f"处理检测结果时出错: {str(e)}")
            
            # 更新跟踪状态
            results_last = current_results
                
            # 处理结果并渲染到帧上
            try:
                display_frame = process_pose_results(frame, raw_results, call_args)
            except Exception as e:
                print(f"处理姿态结果时出错: {str(e)}")
                display_frame = frame.copy()  # 降级为原始帧
            
            # 计算FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 每100帧重置计数器，防止数值过大
            if frame_count >= 100:
                start_time = time.time()
                frame_count = 0
                
            # 获取平均推理时间
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            
            # 添加FPS和性能信息
            avg_inference_time_ms = avg_inference_time * 1000  # 转换为毫秒
            info_text = f"CAM {camera_id} | FPS: {fps:.1f} | Inference: {avg_inference_time_ms:.0f}ms"
            cv2.putText(display_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 将结果存储在共享字典中
            # 优化：使用更低的JPEG质量，进一步加快编码速度
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return_dict[f'frame_{camera_id}'] = buffer.tobytes()
            
            # 主动释放不需要的大型对象和清理内存
            del raw_results
            if 'cuda' in device:
                # 定期清理CUDA缓存
                if frame_count % 100 == 0:
                    torch.cuda.empty_cache()
            
            # 为了调试输出检测到的人数
            if len(current_results) > 0:
                print(f"摄像头 {camera_id} 检测到 {len(current_results)} 人")
            
    except Exception as e:
        print(f"进程 {camera_id} 出错: {str(e)}")
        return_dict[f'error_{camera_id}'] = str(e)
        
    finally:
        # 释放资源
        if 'cap' in locals():
            cap.release()
        print(f"进程 {camera_id} 已退出") 