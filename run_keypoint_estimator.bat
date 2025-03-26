@echo off
echo 关键点估算与标注工具（摄像头版）
echo ============================

REM 检查Python环境
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请确保已安装Python并添加到环境变量
    goto :end
)

REM 设置参数
set CAMERA_ID=0
set OUTPUT_DIR=output_images
set DEVICE=cpu
set DEBUG_MODE=--debug

echo 可用选项:
echo 1. 正常模式（CPU）
echo 2. 调试模式（CPU）
echo 3. 正常模式（CUDA，如果可用）
echo 4. 调试模式（CUDA，如果可用）
echo 5. 退出
echo.

set /p OPTION=请选择运行模式 [1-5]: 

if "%OPTION%"=="1" (
    set DEVICE=cpu
    set DEBUG_MODE=
) else if "%OPTION%"=="2" (
    set DEVICE=cpu
    set DEBUG_MODE=--debug
) else if "%OPTION%"=="3" (
    set DEVICE=cuda:0
    set DEBUG_MODE=
) else if "%OPTION%"=="4" (
    set DEVICE=cuda:0
    set DEBUG_MODE=--debug
) else if "%OPTION%"=="5" (
    goto :end
) else (
    echo 无效选项，使用默认设置（调试模式/CPU）
)

REM 检查输出目录
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo 使用设备: %DEVICE%
echo 摄像头ID: %CAMERA_ID%
echo 输出目录: %OUTPUT_DIR%
if defined DEBUG_MODE echo 调试模式: 已启用
echo.
echo 按任意键开始摄像头实时关键点检测...
echo 运行时按'q'键退出
pause > nul

REM 运行关键点估算程序
python keypoint_estimator.py --camera %CAMERA_ID% --output_dir %OUTPUT_DIR% --device %DEVICE% %DEBUG_MODE%

echo.
if %errorlevel% equ 0 (
    echo 程序已正常退出
) else (
    echo 处理过程中出现错误，请检查日志
)

:end
echo.
echo 按任意键退出...
pause > nul 