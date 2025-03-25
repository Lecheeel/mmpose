import cv2
import torch
import numpy as np
import sys
import os
import time
import argparse
from mmpose.apis import inference_topdown, init_model

# 导入自定义模块
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from multi_camera_pose_estimation.custom_dual_camera import compute_custom_keypoints, draw_custom_keypoints

def parse_args():
    parser = argparse.ArgumentParser(description='静态图像关键点测试')
    parser.add_argument('--image', type=str, required=True, help='图像文件路径')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备，如cuda:0或cpu')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    return parser.parse_args()

def main():
    args = parse_args()
    debug = args.debug
    
    # 检查CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"使用设备: {device}")
    
    # 检查图像文件
    if not os.path.exists(args.image):
        print(f"错误: 找不到图像文件 {args.image}")
        return
    
    # 加载姿态估计模型
    print("加载模型中...")
    model_config = "configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
    model_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
    
    try:
        pose_model = init_model(model_config, model_checkpoint, device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 读取图像
    print(f"读取图像: {args.image}")
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"无法读取图像文件: {args.image}")
        return
    
    print(f"图像尺寸: {frame.shape}")
    
    # 创建输出文件夹
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建显示帧
    display_frame = frame.copy()
    
    # 进行姿态估计
    start_time = time.time()
    print("开始姿态估计...")
    
    try:
        # 姿态估计
        pose_results = inference_topdown(pose_model, frame)
        print(f"姿态估计完成，结果类型: {type(pose_results)}")
        
        if debug:
            print("详细结果信息:")
            print(pose_results)
        
        if pose_results:
            print(f"检测到的人数: {len(pose_results)}")
            
            # 获取第一个人的关键点
            result = pose_results[0]
            print(f"结果类型: {type(result)}")
            
            if debug:
                print("结果详情:")
                print(result)
            
            # 直接从结果中提取关键点和分数
            try:
                keypoints = result.pred_instances.keypoints[0].cpu().numpy()
                keypoint_scores = result.pred_instances.keypoint_scores[0].cpu().numpy()
                print(f"关键点形状: {keypoints.shape}")
                print(f"分数形状: {keypoint_scores.shape}")
                print(f"平均置信度: {np.mean(keypoint_scores):.2f}")
                
                if debug:
                    print("关键点数据:")
                    for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                        print(f"关键点 {i}: 坐标 ({kpt[0]:.1f}, {kpt[1]:.1f}), 置信度: {score:.2f}")
                
                # 绘制原始关键点
                for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                    if score > 0.3:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
                        cv2.putText(display_frame, str(i), (x + 3, y - 3), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 计算自定义关键点
                print("计算自定义关键点...")
                custom_points = compute_custom_keypoints(keypoints, keypoint_scores, thr=0.3)
                print(f"计算了{len(custom_points)}个自定义关键点")
                
                if debug:
                    print("自定义关键点数据:")
                    for key, point in custom_points.items():
                        if isinstance(point, list):
                            print(f"关键点 {key}: 多点列表，包含{len(point)}个点")
                        else:
                            print(f"关键点 {key}: 坐标 {point}")
                else:
                    # 打印部分关键点信息
                    for key in list(custom_points.keys())[:5]:
                        print(f"关键点 {key}: {custom_points[key]}")
                
                # 绘制自定义关键点
                print("绘制自定义关键点...")
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
                if debug:
                    import traceback
                    traceback.print_exc()
        else:
            print("未检测到人体")
            cv2.putText(display_frame, "未检测到人体", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    except Exception as e:
        print(f"姿态估计出错: {e}")
        cv2.putText(display_frame, f"姿态估计错误: {str(e)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if debug:
            import traceback
            traceback.print_exc()
    
    # 保存结果
    print(f"保存结果到: {args.output}")
    cv2.imwrite(args.output, display_frame)
    print(f"处理完成! 结果已保存到 {args.output}")
    
    # 显示结果
    cv2.imshow('处理结果', display_frame)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 