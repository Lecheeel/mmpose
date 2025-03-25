@echo off
echo ===================================================
echo 修复版多进程摄像头系统 - 关键点检测调试
echo ===================================================

echo 这个程序使用简化的架构直接显示关键点，帮助诊断问题

REM 检查Python环境
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到Python。请确保已安装Python并添加到PATH中。
    pause
    exit /b 1
)

echo 按'q'键退出程序

REM 使用调试模式运行程序
python multi_camera_pose_estimation/fix_multiprocess_camera.py --device cuda:0 --debug --kpt_thr 0.3

echo 程序已退出
pause 