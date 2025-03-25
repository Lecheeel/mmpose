import sys
import os

def main():
    """主入口函数"""
    # 将当前目录添加到系统路径
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, current_dir)
    
    # 导入并运行多摄像头姿态估计系统
    try:
        # 尝试导入main函数
        from multi_camera_pose_estimation.main import main
        main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("尝试直接运行main.py脚本...")
        # 如果导入失败，则尝试直接运行脚本
        script_path = os.path.join(current_dir, "multi_camera_pose_estimation", "main.py")
        if os.path.exists(script_path):
            # 通过命令行参数传递给子进程
            import subprocess
            cmd_args = sys.argv[1:]  # 获取传递给run_multi_camera.py的所有参数
            subprocess.run([sys.executable, script_path] + cmd_args)
        else:
            print(f"找不到脚本: {script_path}")

if __name__ == '__main__':
    main() 