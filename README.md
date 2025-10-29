# FastMCP 无人机目标侦察与检索服务

> 基于 FastMCP 的无人机目标侦察、检测与检索服务

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)
[![FastMCP](https://img.shields.io/badge/FastMCP-latest-green.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 项目简介

本项目是一个基于 FastAPI 和 FastMCP 构建的无人机目标侦察与检索服务。它提供了一套完整的工具，用于处理无人机航拍影像，包括目标检测、图像裁剪、相似目标检索、历史影像分析等功能。项目通过 Socket.IO 实现实时数据广播，可与前端应用进行高效交互。

## ✨ 主要特性

- 🚁 **无人机任务规划**: 支持无人机侦察区域规划和航线规划。
- 🎯 **智能目标检测**: 基于 `YOLOv11` 的高精度目标检测。
- 🔍 **多模态图像检索**:
    - **相似目标检索**: 基于 `MAE` 和 `Milvus` 向量数据库的相似目标检索。
    - **历史影像分析**: 结合 `Milvus` 和 `MySQL` 进行历史影像对比与分析。
- 💬 **自然语言查询**: 支持通过自然语言查询数据库（Text2SQL）。
- 🔄 **实时数据广播**: 基于 `Socket.IO` 的实时数据推送，方便与前端集成。
- ⚡ **异步高性能**: 全面基于 `FastAPI` 的异步实现，支持高并发。
- 🧩 **模块化设计**: 清晰的项目结构，易于扩展和维护。

## 🚀 快速开始

### 环境要求

- Python 3.12+
- Uv (用于依赖管理)
- Milvus (向量数据库)
- MySQL (关系型数据库)

### 部署方式

#### 方式一：Docker 部署（推荐）

使用 Docker 可以快速部署包含所有依赖的完整环境：

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

详细的 Docker 部署说明请参考：[Docker 部署指南](DOCKER_DEPLOYMENT.md)

#### 方式二：本地部署

### 安装

1.  **克隆项目**
    ```bash
    git clone <repository-url>
    cd FastMCP_uavTools_v1.3
    ```

2.  **安装依赖**
    项目使用 `uv`进行包管理。
    ```bash
    uv sync
    ```

### 配置

1.  **环境变量**: 创建 `.env` 文件并根据需要配置数据库连接等信息。
2.  **配置文件**: 修改 `configs/config.yaml` 文件，配置模型路径、API端口等参数。

### 激活环境

- linux环境
```bash
source .venv/bin/activate
```

- windows环境
```bash
.venv/Scripts/activate
```

### 启动服务

```bash
uv run api_server.py
```
或者使用 `uvicorn`：
```bash
uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload
```
服务启动后，可以访问 [http://localhost:5000/docs](http://localhost:5000/docs) 查看 API 文档。

## 🏗️ 项目结构

```
FastMCP_uavTools_v1.3/
├── api_server.py               # FastAPI 应用主文件 (包含MCP服务) ⭐
├── config_manager.py           # 配置文件管理器
├── pyproject.toml              # 项目配置文件 (依赖管理)
├── README.md                   # 本文档
├── configs/                    # 配置文件目录
│   └── config.yaml
├── data/                       # 数据文件
│   ├── UAV_images/
│   ├── RS_images/
│   └── ...
├── results/                    # 结果输出目录
│   ├── predicts/               # 预测结果
│   ├── objects/                # 切分的目标图像
│   ├── objects_search/         # 目标检索结果
│   ├── history_search/         # 历史检索结果
│   └── uav_way/                # 无人机路径规划
├── src/                        # 业务逻辑源码
│   ├── img_predictor.py        # 图像预测
│   ├── img_cropper.py          # 图像裁剪
│   ├── plan_uav_route.py       # 无人机路径规划
│   ├── mae_search_image.py     # 目标检索
│   └── ...
└── utils/                      # 工具函数
    ├── milvus_utils.py         # Milvus 数据库工具
    ├── mysql_utils.py          # MySQL 数据库工具
    └── ...
```

## 🔧 技术栈

- **Web 框架**: FastAPI
- **MCP 服务**: FastApiMCP
- **实时通信**: Socket.IO
- **异步服务**: Uvicorn
- **目标检测**: YOLOv11 (via `ultralytics`)
- **图像检索**: MAE, Milvus
- **数据库**: MySQL, Milvus
- **图像处理**: Pillow, OpenCV
- **自然语言处理**: LangChain (for TextSQL)
- **依赖管理**: Uv

## API & MCP 工具

`api_server.py` 中包含了所有的 API 端点和 MCP 工具。具体的功能点请参考代码和 API 文档。

### 主要 MCP 工具

- `uav_trigger`: 无人机区域绘制
- `uav_planner`: 无人机路径规划
- `img_predictor`: 图像目标检测
- `img_cropper`: 图像目标裁剪
- `objects_searcher`: 相似目标图像搜索
- `history_searcher`: 历史目标图像搜索
- `analyze_router`: 车辆路径分析
- ... 以及其他数据查询和广播工具。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

---

**开发者**: Augment Agent  
**最后更新**: 2025-10-24
