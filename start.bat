@echo off
chcp 65001 >nul

REM ========================================
REM 项目环境启动脚本（Windows）
REM 使用方法：
REM   1. 直接启动API服务: start.bat (或 start.bat run)
REM   2. 仅激活环境: start.bat shell
REM   3. 运行其他脚本: start.bat <script.py>
REM ========================================

setlocal EnableDelayedExpansion

REM 获取脚本所在目录作为项目根目录
set "PROJECT_ROOT=%~dp0"
REM 移除路径末尾的反斜杠
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM 将项目根目录添加到 PYTHONPATH
echo "%PYTHONPATH%" | findstr /C:"%PROJECT_ROOT%" >nul
if errorlevel 1 (
    set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
)

REM 打印环境信息用于调试
echo ==========================================
echo 🚀 FastMCP UAV Tools - 环境启动
echo ==========================================
echo 📂 项目根目录: %PROJECT_ROOT%
echo 🐍 PYTHONPATH: %PYTHONPATH%
echo.

REM 检查虚拟环境是否存在，不存在则自动安装依赖
if not exist "%PROJECT_ROOT%\.venv" (
    echo ⚠️  未找到虚拟环境目录 .venv
    echo 🔧 开始自动安装项目依赖...
    echo ==========================================
    echo.
    
    REM Step 1: 安装基础依赖
    echo 📦 Step 1/2: 安装基础依赖...
    uv sync
    if errorlevel 1 (
        echo ❌ 基础依赖安装失败！
        echo 请检查 uv 是否正确安装，或手动运行: uv sync
        pause
        exit /b 1
    )
    echo ✅ 基础依赖安装完成
    echo.
    
    REM Step 2: 安装 PyTorch CPU 版本
    echo 🔥 Step 2/2: 安装 PyTorch CPU 版本...
    uv pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cpu --upgrade
    if errorlevel 1 (
        echo ❌ PyTorch CPU 版本安装失败！
        pause
        exit /b 1
    )
    echo ✅ PyTorch CPU 版本安装完成
    echo.
    
    REM 验证安装
    echo 🔍 验证 PyTorch 安装...
    uv run python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    echo.
    echo ✅ 项目依赖安装完成！
    echo ==========================================
    echo.
)

REM 激活uv虚拟环境
if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    echo 🔄 激活虚拟环境...
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
    echo ✅ 虚拟环境已激活
    echo    Python路径: 
    where python
    echo    Python版本: 
    python --version
) else (
    echo ❌ 虚拟环境激活失败！
    echo 请检查 .venv 目录是否正确创建
    pause
    exit /b 1
)

echo ==========================================
echo.

REM 检查是否提供了参数来运行特定脚本
if "%~1"=="shell" (
    REM 仅激活环境，保持命令行窗口打开供用户使用
    echo ✅ 环境配置完成！您现在可以运行Python命令了。
    echo.
    echo 💡 提示：
    echo    - 运行API服务: uv run api_server.py
    echo    - 退出环境: exit
    echo ==========================================
    echo.
    REM 保持命令行窗口打开
    cmd /k
) else if "%~1"=="run" (
    REM 运行API服务
    echo ▶️  启动API服务...
    echo ==========================================
    echo.
    uv run api_server.py
) else if "%~1"=="" (
    REM 没有参数，默认启动API服务
    echo ▶️  启动API服务（默认行为）...
    echo 💡 提示：如需仅激活环境，请使用 start.bat shell
    echo ==========================================
    echo.
    uv run api_server.py
) else (
    REM 运行指定的Python脚本或命令
    echo ▶️  执行命令: uv run %*
    echo ==========================================
    echo.
    uv run %*
)

endlocal

