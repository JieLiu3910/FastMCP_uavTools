import os
import time
import torch
import numpy as np
from PIL import Image
import json
from transformers import CLIPProcessor, CLIPModel

# Milvus 相关导入
from pymilvus import (
    connections,
    FieldSchema, 
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# 设置处理模式参数: 
# 1 = 重新处理并更新所有目标（包括已存在的）
# 0 = 跳过已存在的目标，只处理新目标
ALWAYS_PROCESS = 0

# Milvus 连接
try:
    connections.connect(host="127.0.0.1", port="19530")
    print("成功连接到 Milvus")
except Exception as e:
    print(f"连接 Milvus 失败: {e}")
    exit(1)

# 设备与模型
# ROOT_DIR = "/mnt/ht3_nas/agent_project/agent5/MCPtool/FastMCP_uavTools_v1.3/data/fleet_999_target"
# ROOT_DIR = "/mnt/ht3_nas/dataset/vehicle_dataset/near_vehi_all_together"

# ROOT_DIR = "./data/missile_boat_target"
# collection_name = "TW_0829_target"

ROOT_DIR = "./data/event_data/20250829_TW/fleet_054A_547_target"
collection_name = "clip_target"


# 设备与模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_local_path = "./model/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name_or_local_path).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

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
vector_dimension = 512  # CLIP-vit-base-patch32 的输出维度
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),  # JSON 字段
]
schema = CollectionSchema(fields=fields, description="Image vectors")

# Milvus 集合操作
# collection_name = "clip_target"

if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
else:
    collection = Collection(name=collection_name, schema=schema)

if not collection.has_index():
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

# def get_geo_info(img_name, img_width, img_height):
#     """从geojson文件中获取地理信息"""
#     try:
#         # 构造geojson文件名（假设文件名规则）
#         geojson_name = img_name.rsplit('.', 1)[0] + '.json'
#         geojson_path = os.path.join(geojson_directory, geojson_name)
        
#         if os.path.exists(geojson_path):
#             with open(geojson_path, 'r') as f:
#                 geo_data = json.load(f)
            
#             # 获取地理边界
#             min_lon = geo_data.get("min_longitude", 0.0)
#             max_lon = geo_data.get("max_longitude", 0.0)
#             min_lat = geo_data.get("min_latitude", 0.0)
#             max_lat = geo_data.get("max_latitude", 0.0)

#             # 计算经纬度与像素的映射关系
#             lon_per_pixel = (max_lon - min_lon) / img_width if img_width > 0 else 0
#             lat_per_pixel = (max_lat - min_lat) / img_height if img_height > 0 else 0
            
#             # 解析时间戳
#             capture_time = geo_data.get("capture_time", "1970-01-01T00:00:00.000Z")
#             try:
#                 import datetime
#                 # 解析ISO格式时间
#                 dt = datetime.datetime.fromisoformat(capture_time.replace('Z', '+00:00'))
#                 timestamp = int(dt.timestamp())
#             except:
#                 timestamp = int(time.time())
            
#             return {
#                 "min_longitude": min_lon,
#                 "max_longitude": max_lon,
#                 "min_latitude": min_lat,
#                 "max_latitude": max_lat,
#                 "lon_per_pixel": lon_per_pixel,
#                 "lat_per_pixel": lat_per_pixel,
#                 "center_longitude": (min_lon + max_lon) / 2,
#                 "center_latitude": (min_lat + max_lat) / 2,
#                 "timestamp": timestamp
#             }
#     except Exception as e:
#         print(f"读取地理信息失败 {img_name}: {e}")
    
#     # 默认值
#     return {
#         "min_longitude": 0.0,
#         "max_longitude": 0.0,
#         "min_latitude": 0.0,
#         "max_latitude": 0.0,
#         "lon_per_pixel": 0.0,
#         "lat_per_pixel": 0.0,
#         "center_longitude": 0.0,
#         "center_latitude": 0.0,
#         "timestamp": int(time.time())
#     }

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

# def calculate_object_geo_position(bbox, geo_info, img_width, img_height):
#     """根据目标bbox中心点计算其地理坐标"""
#     left, top, right, bottom = bbox
    
#     # 计算bbox中心点
#     center_x = (left + right) / 2
#     center_y = (top + bottom) / 2
    
#     # 根据像素位置计算地理坐标
#     # 经度: 从左到右增加
#     object_longitude = geo_info["min_longitude"] + center_x * geo_info["lon_per_pixel"]
    
#     # 纬度: 从上到下减少（图像坐标系y轴向下为正，地理坐标系y轴向上为正）
#     object_latitude = geo_info["max_latitude"] - center_y * geo_info["lat_per_pixel"]
    
#     return object_longitude, object_latitude


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
        
        # # 获取地理信息
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
                    if ALWAYS_PROCESS == 0:
                        print(f"目标 {obj_id} 已存在，跳过处理")
                        skipped_count += 1
                        continue
                    elif ALWAYS_PROCESS == 1:
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

                # 处理图片并生成向量
                try:
                    inputs = processor(images=cropped_image, return_tensors="pt")
                except Exception as proc_error:
                    # 如果仍然出错，尝试使用 do_resize 和 size 参数
                    print(f"使用默认 processor 参数处理 {obj_id} 出错，尝试替代方案...")
                    inputs = processor(
                        images=cropped_image, 
                        return_tensors="pt", 
                        padding=True,
                        do_resize=True,
                        size={"shortest_edge": 224}  # CLIP 模型的标准输入尺寸
                    )
                
                pixel_values = inputs["pixel_values"].to(device)

                feats = model.get_image_features(pixel_values=pixel_values)  # shape [1, 512]
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)    # 为 COSINE 做 L2 归一化

                embedding_array = feats[0].detach().cpu().numpy().astype(np.float32)

                # # 根据bbox中心点计算目标的精确地理坐标
                # obj_longitude, obj_latitude = calculate_object_geo_position(bbox, geo_info, img_width, img_height)

                # 构造完整路径
                original_image_full_path = os.path.join(image_directory, img_file)
                target_image_full_path = os.path.join(target_image_dir, f"{obj_id}.png")
                
                # 将元数据存为JSON 字段
                metadata = {
                    "label": label,
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

# 验证数据库在Milvus查询时是否正常
try:
    # Milvus 中获取实体数量
    count = collection.num_entities
    print(f"集合中的数据量: {count}")
except Exception as e:
    print("查询失败:", e)