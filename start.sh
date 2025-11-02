#!/bin/bash

# ========================================
# 项目环境启动脚本（Linux/Mac）
# 使用方法：
#   1. 直接启动API服务: ./start.sh (或 ./start.sh run)
#   2. 仅激活环境: source start.sh (或 ./start.sh shell)
#   3. 运行其他脚本: ./start.sh <script.py>
# ========================================

# 获取脚本所在目录作为项目根目录
if [ -n "${BASH_SOURCE[0]}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    PROJECT_ROOT="$(pwd)"
fi

# 将项目根目录添加到 PYTHONPATH (仅当不存在时)
if [[ ":$PYTHONPATH:" != *":${PROJECT_ROOT}:"* ]]; then
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+":$PYTHONPATH"}"
fi

# 打印环境信息用于调试
echo "=========================================="
echo "🚀 FastMCP UAV Tools - 环境启动"
echo "=========================================="
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "🐍 PYTHONPATH: ${PYTHONPATH}"
echo ""

# 检查虚拟环境是否存在，不存在则自动安装依赖
if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "⚠️  未找到虚拟环境目录 .venv"
    echo "🔧 开始自动安装项目依赖..."
    echo "=========================================="
    echo ""
    
    # Step 1: 安装基础依赖
    echo "📦 Step 1/2: 安装基础依赖..."
    if ! uv sync; then
        echo "❌ 基础依赖安装失败！"
        echo "请检查 uv 是否正确安装，或手动运行: uv sync"
        exit 1
    fi
    echo "✅ 基础依赖安装完成"
    echo ""
    
    # Step 2: 安装 PyTorch CPU 版本
    echo "🔥 Step 2/2: 安装 PyTorch CPU 版本..."
    if ! uv pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cpu --upgrade; then
        echo "❌ PyTorch CPU 版本安装失败！"
        exit 1
    fi
    echo "✅ PyTorch CPU 版本安装完成"
    echo ""
    
    # 验证安装
    echo "🔍 验证 PyTorch 安装..."
    uv run python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    echo ""
    echo "✅ 项目依赖安装完成！"
    echo "=========================================="
    echo ""
fi

# 激活uv虚拟环境
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    echo "🔄 激活虚拟环境..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    echo "✅ 虚拟环境已激活"
    echo "   Python路径: $(which python)"
    echo "   Python版本: $(python --version)"
else
    echo "❌ 虚拟环境激活失败！"
    echo "请检查 .venv 目录是否正确创建"
    exit 1
fi

echo "=========================================="

# 检查是否提供了参数来运行特定脚本
if [ "$1" = "shell" ]; then
    # 仅激活环境，保持在交互式shell
    echo "✅ 环境配置完成！您现在可以运行Python命令了。"
    echo ""
    echo "💡 提示："
    echo "   - 运行API服务: uv run api_server.py"
    echo "   - 退出环境: exit"
    echo "=========================================="
    exec bash
elif [ "$1" = "run" ]; then
    # 运行API服务
    echo "▶️  启动API服务..."
    echo "=========================================="
    uv run api_server.py
elif [ $# -gt 0 ]; then
    # 如果提供了参数，则直接运行指定的Python脚本或命令
    echo "▶️  执行命令: uv run $@"
    echo "=========================================="
    uv run "$@"
elif [ "$0" = "${BASH_SOURCE[0]}" ]; then
    # 如果是作为可执行文件运行（./start.sh），默认启动API服务
    echo "▶️  启动API服务（默认行为）..."
    echo "💡 提示：如需仅激活环境，请使用 (linux) source start.sh  或 (windows) ./start.sh shell"
    echo "=========================================="
    uv run api_server.py
else
    # 如果是被source的，则只设置环境
    echo "✅ 环境配置完成！您现在可以运行Python命令了。"
    echo "=========================================="
fi