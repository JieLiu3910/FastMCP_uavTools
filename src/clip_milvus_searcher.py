import sys
import os
from PIL import Image
import torch
import time
import json
import numpy as np
from typing import List, Literal, Optional, Dict
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


def search_image_from_milvus(
    query_image: str, 
    query_type: Literal["history", "target"], 
    config_file: Optional[str] = None
) -> List[Dict]:
    """
    搜索与输入图像匹配的历史图像或目标图像
    
    Args:
        query_image: 需要查询的图像路径
        query_type: 查询类型，"history" 表示历史图像查询，"target" 表示目标图像查询
        config_file: 配置文件路径（可选）
    
    Returns:
        parsed_results: 搜索结果列表
            - id: 图像 ID
            - distance: 向量相似度（距离），越大表明图像越相似
            - metadata: 元数据
    """
    # 加载配置
    configs = load_config(config_file)
    
    # 加载 CLIP 模型
    CLIP_model_path = configs.get("clip_model_path")
    model = CLIPModel.from_pretrained(CLIP_model_path)
    processor = AutoImageProcessor.from_pretrained(CLIP_model_path, use_fast=True)
    
    # Milvus 连接
    try:
        connections.connect(
            host=configs["milvus"]["host"], 
            port=configs["milvus"]["port"]
        )
        print("成功连接到 Milvus")
    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        return []
    
    # 根据不同搜索类型检查集合是否存在
    if query_type == "history":
        collection_name = configs.get("milvus")["collections"]["history"]
        if not utility.has_collection(collection_name):
            print(f"集合 {collection_name} 不存在")
            return []
        
        search_configs = configs.get("history_search_params")
        if not search_configs:
            raise KeyError("在 config.yaml 中未找到 'history_search_params' 配置")
    
    elif query_type == "target":
        collection_name = configs.get("milvus")["collections"]["target"]
        if not utility.has_collection(collection_name):
            print(f"集合 {collection_name} 不存在")
            return []
        
        search_configs = configs.get("target_search_params")
        if not search_configs:
            raise KeyError("在 config.yaml 中未找到 'target_search_params' 配置")
    
    else:
        raise ValueError(f"不支持的查询类型: {query_type}，请使用 'history' 或 'target'")
    
    # 获取集合
    collection = Collection(name=collection_name)
    collection.load()
    print(f"成功加载集合 {collection_name}")
    
    # 处理图片生成向量
    start = time.time()
    with open(query_image, "rb") as f:
        query_image_data = Image.open(f).convert("RGB").copy()
    with torch.no_grad():
        inputs = processor(images=query_image_data, return_tensors="pt")
        image_features = model.get_image_features(inputs.pixel_values)
        query_vector = image_features[0].detach().cpu().numpy().astype(np.float32)
    
    # 执行向量搜索，使用 COSINE 度量
    # Milvus 会返回与查询向量相似度在指定区间内的向量
    # 并在 IVF 类索引下会搜索指定数量的聚类中心
    search_params = {
        "metric_type": search_configs["metric_type"],
        "radius": search_configs["radius"],
        "range_filter": search_configs["range_filter"],
        "params": {"nprobe": search_configs["nprobe"]},
    }
    
    # 搜索 Top N 相似向量
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=search_configs["limit"],
        output_fields=["id", "metadata"],  # 返回 id 和 metadata 字段
    )
    
    print(f"{query_type} 图像检索耗时: {time.time() - start:.2f} 秒")
    
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
    # 测试历史图像搜索
    print("💯 ===   测试CLIP历史图像搜索  ===")
    configs = load_config()
    image_file = r"data/event_data/202510_peace_excercise/fleet_055_999_history/images/999_01.jpg"
    
    if os.path.exists(image_file):
        results = search_image_from_milvus(
            query_image=image_file,
            query_type="history"
        )
        
        print("📊 ===  CLIP历史图像搜索结果 === ")
        pprint(results)
        
        print("\n" + "="*50 + "\n")
        
        # 测试目标图像搜索
        print("💯 ===   测试CLIP目标图像搜索  ===")
        results = search_image_from_milvus(
            query_image=image_file,
            query_type="target"
        )
        
        print("📊 ===  CLIP目标图像搜索结果 === ")
        pprint(results)
    else:
        print(f"测试图像文件不存在: {image_file}")

