# from ast import List
import os
import time
from typing import Literal, Optional, List
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import torchvision.transforms as transforms

# Milvus 相关导入
from pymilvus import (
    connections,
    FieldSchema, 
    CollectionSchema,
    DataType,
    Collection,
    utility
)

import sys

# 自建代码导入
current_dir = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件的绝对路径
project_root = os.path.dirname(current_dir)    # 获取项目根目录的路径(假设 utils 是项目根目录下的子目录)
sys.path.append(project_root)   # 将项目根目录动态添加到 Python 搜索路径
from utils.mae_embedding_basic import load_model, InternVisionConfig    # 从 utils.embedding_test 模块中导入 load_model 函数 


def get_objects_from_labelme(img_name):
    """从labelme标注文件中获取目标信息"""
    objects = []
    try:
        # 构造labelme json文件名
        json_name = img_name.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(labelme_json_directory, json_name)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                label_data = json.load(f)
            
            # 遍历所有标注形状
            for i, shape in enumerate(label_data.get("shapes", [])):
                label = shape.get("label", "unknown")
                points = shape.get("points", [])
                
                if len(points) >= 2:
                    # 提取左上和右下坐标
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # 确保坐标顺序正确
                    left = min(x1, x2)
                    top = min(y1, y2)
                    right = max(x1, x2)
                    bottom = max(y1, y2)
                    
                    # 去除文件名中的扩展名
                    img_name_without_ext = img_name.rsplit('.', 1)[0]
                    objects.append({
                        "label": label,
                        "bbox": [left, top, right, bottom],
                        "id": f"{img_name_without_ext}_{label}_{i}"
                    })
    except Exception as e:
        print(f"读取标注信息失败 {img_name}: {e}")
    
    return objects



def embed_history(
    collection: str,
    sorted_img_files: List[str],
    image_directory: str,
    target_image_dir: str,
    labelme_json_directory: str,
    geojson_directory: str,
    model,
    transform,
    device: Literal["cuda","cpu"] = "cpu",
    always_process=1
):
    """
    处理历史图像并生成向量嵌入存储到 Milvus
    
    Args:
        collection: Milvus 集合对象
        sorted_img_files: 排序后的图片文件列表
        image_directory: 原始图片目录路径
        target_image_dir: 目标裁剪图片保存目录路径
        labelme_json_directory: labelme 标注文件目录路径
        geojson_directory: 地理信息文件目录路径
        model: MAE 模型
        transform: 图像预处理转换器
        device: 运行设备 (cuda/cpu)
        always_process: 处理模式，1=重新处理所有目标，0=跳过已存在的目标
    
    Returns:
        dict: 包含处理统计信息的字典
        {
            "processed_count": 新增处理的目标数量,
            "skipped_count": 跳过的目标数量,
            "total_images": 总图片数量
        }
    """
    skipped_count = 0
    processed_count = 0
    
    with torch.no_grad():
        for idx, img_file in enumerate(sorted_img_files, start=1):
            print(f"{idx}: {img_file}")
            start = time.time()
            
            # 打开原始图片以获取尺寸
            try:
                image_path = os.path.join(image_directory, img_file)
                original_image = Image.open(image_path).convert("RGB")
                img_width, img_height = original_image.size
            except Exception as e:
                print(f"打开图片 {img_file} 失败:", e)
                continue
            
            # 获取地理信息
            # geo_info = get_geo_info(img_file, img_width, img_height)
            
            # 获取目标信息
            objects = get_objects_from_labelme(img_file)
            if not objects:
                print(f"图片 {img_file} 没有找到标注目标，跳过处理")
                skipped_count += 1
                continue

            # 处理每个目标
            for obj in objects:
                obj_id = obj["id"]
                label = obj["label"]
                bbox = obj["bbox"]   
                
                # 检查milvus中是否已存在该目标的向量
                try:
                    # 使用 Milvus 查询 id 字段是否存在
                    expr = f"id == {repr(obj_id)}"
                    res = collection.query(expr=expr, output_fields=["id"])
                    if res:
                        if always_process == 0:
                            print(f"目标 {obj_id} 已存在，跳过处理")
                            skipped_count += 1
                            continue
                        elif always_process == 1:
                            print(f"目标 {obj_id} 已存在，但根据设置仍会重新处理并更新")
                except Exception as e:
                    print(f"查询目标是否存在时出错: {e}")
                
                # 裁剪子图像
                try:
                    left, top, right, bottom = map(int, bbox)
                    cropped_image = original_image.crop((left, top, right, bottom))
                    # 保存裁剪后的图像到 target-image 目录
                    target_image_path = os.path.join(target_image_dir, f"{obj_id}.png")
                    cropped_image.save(target_image_path)

                    # 使用 MAE 模型处理图片并生成向量
                    # 应用图像预处理
                    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
                    
                    # 使用 MAE 模型生成向量
                    feats = model(img_tensor)  # shape [1, 1, 3200]
                    
                    # 提取向量并进行 L2 归一化
                    feats = feats.squeeze(1)  # shape [1, 3200]
                    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)  # 为 COSINE 做 L2 归一化

                    embedding_array = feats[0].detach().cpu().numpy().astype(np.float32)

                    print(f"embedding_array:{embedding_array}")
                    print(f"shape:{embedding_array.shape}")
                    print(f"\ntype:{type(embedding_array)}")

                    # 根据bbox中心点计算目标的精确地理坐标
                    # obj_longitude, obj_latitude = calculate_object_geo_position(bbox, geo_info, img_width, img_height)

                    # 构造完整路径
                    original_image_full_path = os.path.join(image_directory, img_file)
                    target_image_full_path = os.path.join(target_image_dir, f"{obj_id}.png")
                    
                    # 将元数据存为JSON 字段
                    metadata = {
                        "label": label,
                        # "timestamp": geo_info["timestamp"],
                        # "latitude": obj_latitude,  # 计算的目标精确纬度
                        # "longitude": obj_longitude,  # 计算的目标精确经度
                        "image_id": img_file,
                        "original_image_path": original_image_full_path,  # 原始图片完整路径
                        "target_image_path": target_image_full_path  # 目标图片完整路径
                    }

                    # 存储到milvus
                    data = [
                        [obj_id],  # id
                        [embedding_array.tolist()],  # vector
                        [json.dumps(metadata, ensure_ascii=False)],  # metadata (JSON)
                    ]                
                    try:
                        collection.insert(data)
                        processed_count += 1
                        print(f"处理目标: {obj_id}")
                    except Exception as e:
                        print(f"插入数据到 Milvus 失败 {obj_id}: {e}")
                        continue

                except Exception as e:
                    print(f"处理目标 {obj_id} 时出错:", e)
                    continue
            
            end = time.time()
            print(f"处理图片 {img_file} 耗时: {end - start:.2f} 秒")

    print(f"所有图片处理完成。新增 {processed_count} 个目标的向量，跳过 {skipped_count} 个已存在的目标")
    
    return {
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "total_images": len(sorted_img_files)
    }

if __name__ == "__main__":
    # 设置处理模式参数: 
    # 1 = 重新处理并更新所有目标（包括已存在的）
    # 0 = 跳过已存在的目标，只处理新目标

    ALWAYS_PROCESS = 1

    # Milvus 连接
    try:
        connections.connect(host="10.200.50.4", port="19530")
        print("成功连接到 Milvus")
    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        exit(1)

    # 设备与模型
    ROOT_DIR = "/mnt/ht3_nas/agent_project/agent4/MCPtool/FastMCP_uavTools_v1.3/data/fleet_999_history"

    # 设备与模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/mnt/ht3_nas/group/lj/model/saved_internvit_model.pth"


    # 加载 MAE 模型
    model = load_model(model_path, device=device)
    # model = torch.load(model_path, map_location=device, weights_only=False)

    # model = model.to(torch.bfloat16)  # 使用 bfloat16 精度
    model.eval()

    # 定义图像预处理转换器
    # 根据 embedding_test.py 中的示例，使用特定的均值和标准差
    mean = [86.82476806640625, 92.53337097167969, 89.27667236328125]
    std = [106.50431060791016, 109.83255767822266, 116.78082275390625]

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),  # 转换回 0-255 范围
        transforms.Normalize(mean=mean, std=std)
    ])


    # 图片目录和支持的格式
    image_directory = os.path.join(ROOT_DIR, "images")
    supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    img_files = [f for f in os.listdir(image_directory) if f.lower().endswith(supported_extensions)]
    sorted_img_files = sorted(img_files)

    # 标注文件和地理信息文件路径
    labelme_json_directory = os.path.join(ROOT_DIR, "labelme_json")
    geojson_directory = os.path.join(ROOT_DIR, "geojson")

    # 创建提取到的目标小图文件存储路径
    target_image_dir = os.path.join(ROOT_DIR,"target_image")
    os.makedirs(target_image_dir, exist_ok=True)

    # 定义向量索引的 Schema
    vector_dimension = 1024  # CLIP-vit-base-patch32 的输出维度
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),  # JSON 字段
    ]

    schema = CollectionSchema(fields=fields, description="Image vectors")

    # Milvus 集合操作
    collection_name = "fleet_target"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    collection = Collection(name=collection_name, schema=schema)

    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    print("向量索引创建完成")

    # 批量处理图片并导入向量
    skipped_count = 0
    processed_count = 0


    # 调用函数进行处理
    result = embed_history(
        collection=collection,
        sorted_img_files=sorted_img_files,
        image_directory=image_directory,
        target_image_dir=target_image_dir,
        labelme_json_directory=labelme_json_directory,
        geojson_directory=geojson_directory,
        model=model,
        transform=transform,
        device=device,
        always_process=ALWAYS_PROCESS
    )

    print(f"\n处理结果汇总:")
    print(f"  - 处理的目标数: {result['processed_count']}")
    print(f"  - 跳过的目标数: {result['skipped_count']}")
    print(f"  - 总图片数: {result['total_images']}")

    # 验证数据库在Milvus查询时是否正常
    try:
        # Milvus 中获取实体数量
        count = collection.num_entities
        print(f"集合中的数据量: {count}")
    except Exception as e:
        print("查询失败:", e)