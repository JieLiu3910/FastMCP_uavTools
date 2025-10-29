#!/bin/bash
set -e

echo "=========================================="
echo "FastMCP UAV Tools Service Starting..."
echo "=========================================="

# Check if config file exists
if [ ! -f "/app/configs/config.yaml" ]; then
    echo "WARNING: config.yaml not found, using Docker config example"
    if [ -f "/app/configs/config.docker.yaml" ]; then
        cp /app/configs/config.docker.yaml /app/configs/config.yaml
    else
        echo "ERROR: Configuration file does not exist"
        exit 1
    fi
fi

# Check YOLO model file
if [ ! -f "/app/configs/yolo11_weights_best.pt" ]; then
    echo "WARNING: YOLO model file does not exist: /app/configs/yolo11_weights_best.pt"
    echo "Please ensure the model file is properly mounted"
fi

# Wait for Milvus service to be ready (if using docker-compose)
if [ -n "$MILVUS_HOST" ]; then
    echo "Waiting for Milvus service to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if nc -z "$MILVUS_HOST" "${MILVUS_PORT:-19530}" 2>/dev/null; then
            echo "Milvus service is ready"
            break
        fi
        echo "Waiting for Milvus to start... ($timeout seconds)"
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "WARNING: Milvus connection timeout, continuing startup..."
    fi
fi

# Wait for MySQL service to be ready (if using docker-compose)
if [ -n "$MYSQL_HOST" ]; then
    echo "Waiting for MySQL service to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if nc -z "$MYSQL_HOST" "${MYSQL_PORT:-3306}" 2>/dev/null; then
            echo "MySQL service is ready"
            break
        fi
        echo "Waiting for MySQL to start... ($timeout seconds)"
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "WARNING: MySQL connection timeout, continuing startup..."
    fi
fi

echo "=========================================="
echo "Starting FastAPI application..."
echo "=========================================="

# Execute the passed command
exec "$@"

