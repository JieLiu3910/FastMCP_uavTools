# Docker 构建说明


## 快速构建命令

### 构建镜像

```bash
# 基础构建
docker build -t uavtools:v1.1 .

# 带构建参数
docker build --build-arg PYTHON_VERSION=3.13 -t uavtools:latest .

# 不使用缓存重新构建
docker build --no-cache -t uavtools:latest .
```

### 使用 docker-compose

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 仅构建镜像（不启动）
docker-compose build

# 查看构建日志
docker-compose build --progress=plain
```

## 构建优化建议

### 1. 使用构建缓存

Dockerfile 已按照依赖变化频率进行了层级优化：

```dockerfile
COPY pyproject.toml uv.lock ./  # 依赖文件单独复制
RUN uv sync --frozen --no-cache  # 利用层缓存
COPY . .                          # 最后复制代码
```

### 2. 多架构构建

如需支持 ARM64 架构（如 Apple M1/M2）：

```bash
# 使用 buildx 构建多架构镜像
docker buildx build --platform linux/amd64,linux/arm64 -t fastmcp-uavtools:latest .
```

### 3. 构建时优化

```bash
# 限制构建资源
docker build --memory=4g --cpus=2 -t fastmcp-uavtools:latest .
```

## 镜像大小优化

当前镜像大小约 **3-4GB**（包含深度学习依赖）。

### 优化建议：

1. **使用 slim 基础镜像** ✅ 已使用 `python:3.13-slim`
2. **清理 apt 缓存** ✅ 已在 Dockerfile 中清理
3. **模型文件外置** ⭐ 推荐通过挂载卷提供大模型文件

## 生产环境构建

### 1. 设置版本标签

```bash
# 使用 Git commit hash
docker build -t fastmcp-uavtools:$(git rev-parse --short HEAD) .

# 使用版本号
docker build -t fastmcp-uavtools:v1.3.0 .
```

### 2. 推送到镜像仓库

```bash
# Docker Hub
docker tag fastmcp-uavtools:latest yourusername/fastmcp-uavtools:latest
docker push yourusername/fastmcp-uavtools:latest

# 私有仓库
docker tag fastmcp-uavtools:latest registry.example.com/fastmcp-uavtools:latest
docker push registry.example.com/fastmcp-uavtools:latest
```

## 故障排查

### 构建失败

1. **UV 安装失败**
   ```bash
   # 检查网络连接
   docker build --network=host -t fastmcp-uavtools:latest .
   ```

2. **依赖安装超时**
   ```bash
   # 增加超时时间
   docker build --build-arg UV_HTTP_TIMEOUT=600 -t fastmcp-uavtools:latest .
   ```

3. **磁盘空间不足**
   ```bash
   # 清理旧镜像
   docker system prune -a
   ```

### 验证构建

```bash
# 查看镜像信息
docker images fastmcp-uavtools

# 验证镜像层
docker history fastmcp-uavtools:latest

# 测试运行
docker run --rm fastmcp-uavtools:latest uv --version
```

## CI/CD 集成

### GitHub Actions 示例

```yaml
name: Build Docker Image

on:
  push:
    branches: [main]
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ secrets.DOCKER_REGISTRY }}/fastmcp-uavtools:latest
```

### GitLab CI 示例

```yaml
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## 参考资源

- [Docker 最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [UV 文档](https://github.com/astral-sh/uv)
- [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)

