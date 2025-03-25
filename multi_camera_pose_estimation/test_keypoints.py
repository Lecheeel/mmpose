import cv2
import torch
import numpy as np
import sys
import os
import time
from mmpose.apis import inference_topdown, init_model

# 导入自定义模块
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from multi_camera_pose_estimation.custom_dual_camera import compute_custom_keypoints, draw_custom_keypoints

def main():
    # 检查CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载姿态估计模型
    print("加载模型中...")
    model_config = "configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
    model_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
    
    try:
        pose_model = init_model(model_config, model_checkpoint, device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("开始测试，按'q'退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        # 创建显示帧
        display_frame = frame.copy()
        
        # 进行姿态估计
        start_time = time.time()
        print("\n------ 新帧处理 ------")
        
        try:
            # 姿态估计
            pose_results = inference_topdown(pose_model, frame)
            print(f"姿态估计完成，结果类型: {type(pose_results)}")
            
            if pose_results:
                print(f"检测到的人数: {len(pose_results)}")
                
                # 获取第一个人的关键点
                result = pose_results[0]
                print(f"结果类型: {type(result)}")
                
                # 直接从结果中提取关键点和分数
                try:
                    keypoints = result.pred_instances.keypoints[0].cpu().numpy()
                    keypoint_scores = result.pred_instances.keypoint_scores[0].cpu().numpy()
                    print(f"关键点形状: {keypoints.shape}")
                    print(f"分数形状: {keypoint_scores.shape}")
                    print(f"平均置信度: {np.mean(keypoint_scores):.2f}")
                    
                    # 绘制原始关键点
                    for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                        if score > 0.3:
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
                            cv2.putText(display_frame, str(i), (x + 3, y - 3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 计算自定义关键点
                    custom_points = compute_custom_keypoints(keypoints, keypoint_scores, thr=0.3)
                    print(f"计算了{len(custom_points)}个自定义关键点")
                    
                    # 打印部分关键点信息
                    for key in list(custom_points.keys())[:5]:
                        print(f"关键点 {key}: {custom_points[key]}")
                    
                    # 绘制自定义关键点
                    display_frame = draw_custom_keypoints(display_frame, custom_points, 
                                                          radius=10, thickness=4)
                    
                    # 显示处理时间
                    process_time = time.time() - start_time
                    cv2.putText(display_frame, f"处理时间: {process_time:.2f}秒", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示关键点数量
                    cv2.putText(display_frame, f"原始关键点: {len(keypoints)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"自定义关键点: {len(custom_points)}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"处理关键点时出错: {e}")
                    cv2.putText(display_frame, f"关键点处理错误: {str(e)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print("未检测到人体")
                cv2.putText(display_frame, "未检测到人体", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"姿态估计出错: {e}")
            cv2.putText(display_frame, f"姿态估计错误: {str(e)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow('测试关键点', display_frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("测试结束")

if __name__ == '__main__':
    main() 