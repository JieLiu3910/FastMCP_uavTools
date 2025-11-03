# ================================
# FastMCP UAV Tools Docker Image
# ================================

# Use official Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - build-essential: Required for compiling Python packages
# - libgl1: OpenCV dependency
# - libglib2.0-0: OpenCV dependency
# - curl: Health check
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV (Python package manager)
# Using official recommended installation method
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files (leverage Docker cache layer)

COPY pyproject.toml uv.lock ./

# Install Python dependencies
# --system: Install to system Python environment instead of virtual environment
# --frozen: Use lockfile without updates
RUN uv sync --frozen --no-cache

# Copy project code
COPY . .

# Copy startup script and set permissions
# COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh /app/start.sh

# Create necessary directories
RUN mkdir -p \
    /app/results/predicts \
    /app/results/photographs \
    /app/results/objects \
    /app/results/objects_search \
    /app/results/history_search \
    /app/results/uav_way \
    /app/results/history_image \
    /app/results/objects_image 


# Mount volumes
# VOLUME ["/app/configs"]
# VOLUME ["/app/data"]
# VOLUME ["/app/model"]
# VOLUME ["/app/results"]

COPY config_manager_docker.py /app/config_manager.py 

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Use startup script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Startup command
# CMD ["uv", "run", "api_server:socket_app", "--host", "0.0.0.0", "--port", "5000"]
# CMD ["uv", "run", "api_server.py"]
CMD ["./start.sh"]

