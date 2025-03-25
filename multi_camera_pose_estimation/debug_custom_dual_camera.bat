@echo off
echo ===================================================
echo 双摄像头姿态估计系统 - 调试模式
echo ===================================================

REM 检查Python环境
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到Python。请确保已安装Python并添加到PATH中。
    pause
    exit /b 1
)

REM 检查CUDA可用性
python -c "import torch; print('CUDA可用:',torch.cuda.is_available())" 

echo 正在调试模式下启动双摄像头系统，请确保已连接两个摄像头...
echo 按'q'键可随时退出程序

REM 使用调试模式运行程序
python multi_camera_pose_estimation/custom_dual_camera_main.py --device cuda:0 --debug

echo 程序已退出
pause 