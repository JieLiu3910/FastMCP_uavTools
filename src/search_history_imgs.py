import sys
import os
from PIL import Image
import torch
import time
import json
import numpy as np
from typing import List, Optional,Dict
from datetime import datetime
from transformers import CLIPModel, AutoImageProcessor
from pprint import pprint

from pymilvus import connections, Collection, utility  # Milvus 相关导入
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_manager import load_config

def search_history_from_milvus(query_image: str, configs: Dict)->List[Dict]:
    """
    搜索与输入图像匹配的历史图像

    Args:
        query_image: 需要查询的图像directory
        configs: 配置字典

    Returns:
        parsed_results: 搜索结果
            - id: 图像 ID
            - distance: 向量相似度（距离），越大表明图像越相似
            - metadata: 元数据
    """

    # 加载 CLIP 模型
    CLIP_model_path = configs.get("clip_model_path")
    model = CLIPModel.from_pretrained(CLIP_model_path)
    processor = AutoImageProcessor.from_pretrained(CLIP_model_path, use_fast=True)

    # Milvus 连接
    try:
        connections.connect(
            host=configs["milvus"]["host"], port=configs["milvus"]["port"]
        )
        print("成功连接到 Milvus")
    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        return []

    # 检查集合是否存在
    # collection_name = "images_vector_history"
    collection_name = configs.get("milvus")["collections"]["history"]
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return []

    # 获取集合
    collection = Collection(name=collection_name)
    collection.load()
    print(f"成功加载集合 {collection_name}")

    # 处理图片生成向量
    start = time.time()
    with open(query_image, "rb") as f:
        query_image_data = Image.open(f).convert("RGB").copy()
    with torch.no_grad():
        inputs = processor(query_image_data, return_tensors="pt")
        image_features = model.get_image_features(inputs.pixel_values)
        query_vector = image_features[0].detach().cpu().numpy().astype(np.float32)

    # 执行向量搜索，使用 COSINE 度量
    # Milvus 会返回与查询向量相似度在 (0.7, 1.0] 区间内的向量
    # 并在 IVF 类索引下会搜索 10 个聚类中心
    search_configs = configs.get("history_search_params")
    if not search_configs:
        raise KeyError("在 config.yaml 中未找到 'history_search_params' 配置")

    search_params = {
        "metric_type": search_configs["metric_type"],
        "radius": search_configs["radius"],
        "range_filter": search_configs["range_filter"],
        "params": {"nprobe": search_configs["nprobe"]},
    }

    # 搜索 Top 10 相似向量
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=search_configs["limit"],
        output_fields=["id", "metadata"],  # 返回 id 和 metadata 字段
    )

    print(f"目标图像检索耗时: {time.time() - start:.2f} 秒")

    # 解析搜索结果
    parsed_results = []
    for hits in results:
        for hit in hits:
            result_item = {
                "id": hit.entity.get("id"),
                "distance": hit.distance,
            }

            # 解析 metadata JSON 字段
            metadata_str = hit.entity.get("metadata")
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                    result_item.update(metadata)
                except json.JSONDecodeError:
                    print(f"解析 metadata 失败: {metadata_str}")

            parsed_results.append(result_item)

    return parsed_results



if __name__ == "__main__":

    # 测试MySQL历史图像搜索
    print("💯 ===   测试MySQL历史图像搜索  ===")
    # 为了能够独立运行，这里我们加载配置
    configs = load_config()
    image_file = r"/mnt/ht3_nas/agent_project/agent5/MCPtool/FastMCP_uavTools_v1.3/data/fleet_999_history/images/999_01.jpg"
    results = search_milvus_history(
        query_image=image_file,
        configs=configs,
    )

    print("📊 ===  MySQL历史图像搜索结果 === ")
    pprint(results)
