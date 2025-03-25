@echo off
echo ===================================================
echo 关键点测试程序 - 诊断模式
echo ===================================================

echo 这个程序将直接测试单摄像头关键点检测，帮助诊断问题

REM 检查Python环境
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到Python。请确保已安装Python并添加到PATH中。
    pause
    exit /b 1
)

REM 运行测试程序
python multi_camera_pose_estimation/test_keypoints.py

echo 测试完成
pause 