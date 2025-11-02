"""
FastAPI应用 - YOLO目标检测服务

运行方式:
    >> python tools_server.py

    >> uvicorn tools_server:app --host 0.0.0.0 --port 8000

API端点:
    POST /predict - 提供图片路径进行目标检测
    POST /crop - 根据检测结果JSON文件裁剪图像中的目标
    GET /health - 健康检查
    静态文件: /results - 访问检测结果图片和裁剪后的图像
"""

# 基础库
import sys
import os
import time
import glob
import shutil
import json
import asyncio
import ast
import uuid
import requests
import socketio
import uvicorn
import traceback

from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Literal, Optional, Union, Dict, Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# 引入热插拔模块
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# API库 & MCP库
from pydantic import BaseModel, Field
from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP


# 导入自定义工具函数
from utils.rs_image_utils import get_satellite_from_mysql
from utils.route_analyzer import analyze_route_main
from utils.mysql_utils import get_field_names_only, init_text2sql,query_image_data, query_equipment_data
from utils.mae_embedding_basic import InternVisionConfig

from src.img_predictor import predict  # 图像预测
from src.img_cropper import main as crop_objects  # 图像裁切
from src.uav_route_planner import uav_tracking_shooting  # 无人机路线规划
from src.target_label_analyzer import analyze_target_label  # 标签分析
from src.mae_milvus_searcher import search_image_from_milvus as mae_searcher 
from src.clip_milvus_searcher import search_image_from_milvus as clip_searcher

# 导入配置文件
from config_manager import load_config  # 配置文件解析
from dotenv import load_dotenv
load_dotenv()



# 全局配置信息
global_config = load_config()

print("=" * 100)
print(f" ⚙️  global_config:\n")
pprint(global_config)
print(f'{"=" * 100}\n')


# ================================== Hot-reloading config ==================================

class ConfigUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event):
        config_file_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
        if not event.is_directory and event.src_path == config_file_path:
            print("🚀 Detected change in config.yaml, reloading configuration...")
            try:
                # Use a temporary variable to load the new config
                new_config = load_config()
                
                # Update the global config dictionary in-place
                global_config.clear()
                global_config.update(new_config)
                
                print("✅ Configuration reloaded successfully.")
                print("=" * 100)
                print(f" ⚙️  New global_config:\n")
                pprint(global_config)
                print(f'{"=" * 100}\n')
                
            except Exception as e:
                print(f"❌ Error reloading configuration: {e}")

def start_config_watcher():
    """Starts a file watcher on the config file in a separate thread."""
    config_path = os.path.join(ROOT_DIR, "configs")
    event_handler = ConfigUpdateHandler()
    observer = Observer()
    observer.schedule(event_handler, config_path, recursive=False)
    
    # Start the observer in a separate thread
    observer_thread = threading.Thread(target=observer.start)
    observer_thread.daemon = True  # Allows main thread to exit
    observer_thread.start()
    
    print(f"👀 Started configuration watcher on '{config_path}'")
	

# =========================================  基本配置信息  =========================================


# 服务端口
PORT = global_config["api_url_port"]
BASE_URL = f"http://localhost:{PORT}"

PHOTOGRAPHS_URL = f"/results/photographs/"
PREDICT_URL = f"/results/predicts/"
CROP_URL = f"/results/objects/"
OBJECTS_URL = f"/results/objects_image/"
HISTORY_URL = f"/results/history_image/"
UAV_URL = f"/results/uav_way/"


# 目录设置
ROOT_DIR = os.path.dirname(__file__)
RESULTS_DIR = Path(os.path.join(ROOT_DIR, "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.path.join(ROOT_DIR, "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# UAV结果目录
PREDICTS_DIR = RESULTS_DIR / "predicts"
PHOTOGRAPHS_DIR = RESULTS_DIR / "photographs"
OBJECTS_DIR = RESULTS_DIR / "objects"
OBJECTS_SEARCH_DIR = RESULTS_DIR / "objects_search"
HISTORY_SEARCH_DIR = RESULTS_DIR / "history_search"
UAV_WAY_DIR = RESULTS_DIR / "uav_way"
HISTORY_IMAGE_DIR = RESULTS_DIR / "history_image"
OBJECTS_IMAGE_DIR = RESULTS_DIR / "objects_image"

# 遥感影像存储路径
RS_IMAGES_DIR = Path(os.path.join(ROOT_DIR, "data", "RS_images"))


# 确保子目录存在
PREDICTS_DIR.mkdir(exist_ok=True)
PHOTOGRAPHS_DIR.mkdir(exist_ok=True)
OBJECTS_DIR.mkdir(exist_ok=True)
OBJECTS_SEARCH_DIR.mkdir(exist_ok=True)
HISTORY_SEARCH_DIR.mkdir(exist_ok=True)
UAV_WAY_DIR.mkdir(exist_ok=True)
HISTORY_IMAGE_DIR.mkdir(exist_ok=True)
OBJECTS_IMAGE_DIR.mkdir(exist_ok=True)
RS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================  定义请求模型  =========================================

class RSImagePushRequest(BaseModel):
    RSImagePushData: Optional[Dict[str, Any]] | List[Dict[str, Any]] = None # 遥感影像元数据

request: Dict[str, Any] | List[Dict[str, Any]]

class strTypeRequest(BaseModel):
    input_str: str
    keyRegion:str =  "霍尔木兹海峡"

# ---------    基础请求模型     ---------
class Position(BaseModel):
    """位置信息模型"""

    lat: float  # 纬度
    lon: float  # 经度
    alt: float  # 高度（米）

class FileListRequest(BaseModel):
    """文件列表请求模型"""
    img_name: str

class BroadcastUAVPointRequest(BaseModel):
    """无人机航点广播请求模型"""
    uav_route_data: Optional[Dict[str, Any]] = None
    num_points: Optional[int] = None
    location_name: Optional[str] = None
    location_longitude: float
    location_latitude: float # 地点名称,如"北京"

class UAVTriggerRequest(BaseModel):
    """无人机触发请求模型"""
    longitude: float
    latitude: float
    radius: float

class UAVImageRequest(BaseModel):
    """无人机图像请求模型"""
    location_name: str
    location_longitude: float
    location_latitude: float # 地点名称,如"北京"

# ---------    图片预测请求模型     ---------

class PredictRequest(BaseModel):
    """图片预测请求模型"""
    image_path: str

class CropRequest(BaseModel):
    """图片裁剪请求模型"""
    predicted_json_path: str


class TargetSearchRequest(BaseModel):
    """相似目标/历史目标图像搜索请求模型"""
    objects_list: List[str]
    id: Optional[List[str]] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    extent: Optional[List[float]] = None
    query_type: Literal["history","target"] = "history"  # 搜索类型: "history" 或 "target"


class RouteAnalysisRequest(BaseModel):
    """车辆路径分析请求模型"""
    history_json_list: List[str]

class LabelAnalysisResult(BaseModel):
    """标签分析结果模型"""

    conclusion: str
    label_distribution: Optional[Dict[str, int]] = None
    details: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"  # 允许额外字段，兼容 analyze_target_label 返回的完整数据

class UAVScanRequest(BaseModel):
    """无人机扫描目标请求模型"""

    uav_pos: Position  # 无人机初始位置
    destination_pos: Position  # 目的地位置
    current_time: str  # 当前时间，格式 "HH:MM:SS"
    scan_mode: Optional[int] = 1  # 模式:[0:直线扫描（默认）； 1：螺旋扫描]


# Pydantic装备数据格式定义
class EquipmentItem(BaseModel):
    # 基础字段
    id: Optional[str] = None
    topic: Optional[str] = None
    layer: Optional[str] = None
    class_name: Optional[str] = Field(
        None, alias="class"
    )  # class是Python关键字，使用alias
    camp: Optional[str] = None

    # 位置信息
    lon: Optional[Union[str, float]] = None
    lat: Optional[Union[str, float]] = None
    high: Optional[Union[str, float]] = None

    # 状态信息
    zone: Optional[str] = None
    status: Optional[str] = None
    ISL_id: Optional[Union[str, List[str]]] = None  # 链路信息

    # 姿态信息
    pitch_angle: Optional[Union[str, float]] = None
    yaw_angle: Optional[Union[str, float]] = None
    roll_angle: Optional[Union[str, float]] = None

    # 速度信息
    velocity_x: Optional[Union[str, float]] = None
    velocity_y: Optional[Union[str, float]] = None
    velocity_z: Optional[Union[str, float]] = None

    # 其他字段（支持任意额外字段）
    class Config:
        extra = "allow"  # 允许额外字段
        populate_by_name = True  # 允许通过别名填充


class GeocodeRequest(BaseModel):
    """地理定位请求模型"""
    position: Dict[str, Any]
    # zoom: Optional[int] = 0


class RSimageRrequest(BaseModel):

    # 卫星网页数据请求参数
    acquisitionTime: List[Dict[str, int]] = None # 影像采集时间 {"Start": start_timestamp,"End": end_timestamp}
    extent: List[float] = None  # bbox[minX(西), minY(北), maxX(东), maxY(南)]
    cloud_percent_min: Optional[int] = 0  # 云量最小值
    cloud_percent_max: Optional[int] = 20  # 云量最大值
    limit: Optional[int] = None  # 限制返回记录数

    # MySQL存储参数
    host: Optional[str] = "localhost"  # 数据库地址
    port: Optional[int] = 3306  # 数据库端口
    user: Optional[str] = "root"  # 数据库用户名
    password: Optional[str] = "123456"  # 数据库密码
    database: Optional[str] = "RS_images_db"  # 数据库名称
    table_name: Optional[str] = "RS_images_metadata"  # 数据库表名

    # class Config:
    #     extra = "forbid"  # 禁止额外字段，确保API调用时参数准确


class EquipmentQueryRequest(BaseModel):
    extent: list[float] = None
    keyRegion: str = None
    topic: str = None
    layer: str = None
    camp: str = None
    status: str = None
    database: str = None
    table_name: str = None
    limit: int = 30


# ======================================================================================================
# =====================================       Socket.IO 服务器       ====================================
# ======================================================================================================

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # 允许所有来源，生产环境中应该限制具体域名
    logger=True,
    engineio_logger=True,
)

# 创建FastAPI实例
app = FastAPI(title="工具API", description="智能体工具服务", version="1.0.0")


# 启动热插拔参数配置服务
@app.on_event("startup")
def startup_event():
    """Initializes resources on application startup."""
    start_config_watcher()


# 在创建 FastAPI 实例后添加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@sio.event
async def connect(sid, environ):
    print(f"🔗 新客户端连接: {sid}")
    return True  # 接受连接

# 客户端加入房间事件
@sio.on("join")
async def join(sid, data):
    room = data.get("room")
    if room:
        sio.enter_room(sid, room)
        print(f"🚪 客户端 {sid} 已加入房间: {room}")

# 客户端离开房间事件
@sio.on("leave")
async def leave(sid, data):
    room = data.get("room")
    if room:
        sio.leave_room(sid, room)
        print(f"👋 客户端 {sid} 已离开房间: {room}")

# 客户端断开连接事件
@sio.event
async def disconnect(sid):
    print(f"🔌 客户端断开连接: {sid}")


# 6. FastAPI-MCP 服务器设置
# 创建Socket.IO ASGI应用
socket_app = socketio.ASGIApp(sio, app)


# ======================================================================================================
# =====================================       FastAPI 路由        ======================================
# ======================================================================================================

@app.get("/", operation_id="root")
async def root():
    """根路径 - API信息"""
    return {
        "message": "YOLO目标检测API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "img_predictor": "POST /predict - 提供图片路径进行目标检测",
            "img_cropper": "POST /crop - 根据检测结果JSON文件裁剪图像中的目标",
            "img_matcher": "POST /discover_related_images - 根据照片文件名匹配相关的预测、目标、搜索结果文件",
            "targets_searcher": "POST /target_image_search - 目标图像搜索与分析",
            "history_searcher": "POST /history_image_search - 历史图像搜索",
            "uav_planner": "POST /tracker - 执行无人机扫描目标计算",
        },
        "static_access": {
            "results": "/results/{directory_name}/{filename} - 直接访问结果文件",
            "directories": [
                "predicts",
                "photographs",
                "objects",
                "objects_search",
                "history_search",
                "uav_way",
                "history_image",
                "objects_image",
            ],
            "rs_images": "/data/RS_images/{filename} - 直接访问遥感影像文件",
        },
        "socketio": {
            "broad_equipData": "POST /broad_equipData - 广播装备数据到Socket.IO客户端",
            "broad_uavData": "POST /broad_uavData - 广播无人机路径数据到Socket.IO客户端（每0.5秒一个点）",
            "broadcast_RSimage": "POST /broadcast_RSimage - 广播遥感影像数据到Socket.IO客户端",
            "rooms": "GET /socketio/rooms - 获取Socket.IO房间信息",
            "manual_broadcast": "POST /socketio/broadcast - 手动广播消息",
        },
        "examples": {
            "discover_related": "POST /discover_related_images - 根据文件名获取相关文件信息",
            "access_predict_result": "GET /results/predicts/TK01(3)_predict_info.json",
            "access_predict_image": "GET /results/predicts/TK01(3)_predict.jpg",
            "access_cropped_object": "GET /results/objects/TK01(3)_tank_1.jpg",
            "access_search_json": "GET /results/objects_search/TK01_target.json",
        },
    }


# ====================================   无人机图像处理工具路由   ====================================

#tag 无人机侦察工具路由

@app.post("/uav_trigger", operation_id="uav_trigger")
async def uav_trigger(request: UAVTriggerRequest):
    """
    启动无人机侦察区域绘制

    Args:
        request: 信息请求体
            - longitude: 无人机侦察区域中心点位置：(longitude,latitude)
            - latitude: 无人机侦察区域中心点位置：(longitude,latitude)
            - radius: 侦察区域半径

    Returns:
        JSONResponse: 包含侦察区域参数的响应.
    """

    try:
        message_id = str(uuid.uuid4())
        print(
            f"收到无人机侦察区域请求: 位置=({request.longitude}, {request.latitude}), 半径={request.radius}km"
        )
        
        broadcast_payload={
            "message": "🚁 无人机侦察区域绘制成功!",
            "message_id": message_id,
            "type": "uavTrigger",
            "timestamp": datetime.now().isoformat(),
            "data_count": 1,
            "data": {
                "longitude": request.longitude,
                "latitude": request.latitude,
                "radius": request.radius
            }
        }

        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        print(f"Error processing UAV detect region request: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start UAV detect task")


@app.post("/uav_planner", operation_id="uav_planner")
async def uav_planner(request: UAVScanRequest):
    """
    执行无人机路线规划

    Args:
        request: 信息请求体
            - uav_pos: 无人机起始位置:(latitude,longitude,altitude)
            - destination_pos: 目的地位置:(latitude,longitude,altitude)
            - current_time: 时间信息
            - scan_mode: 扫描模式, 默认值为0

    Returns:
        JSONResponse: 包含扫描结果的响应
    """
    try:
        message_id = str(uuid.uuid4())
        uav_config = global_config["uav_params"]

        # 解析时间字符串
        print(
            f"🔍 接收到的时间字符串: '{request.current_time}' (type: {type(request.current_time)})"
        )

        try:
            current_time_obj = parse_time_string(request.current_time)
            # current_time_obj = datetime.strptime(request.current_time, "%H:%M:%S").time()
        except ValueError as ve:
            print(f"❌ 时间解析失败: {ve}")
            raise HTTPException(
                status_code=400, detail="时间格式错误，请使用 HH:MM:SS 格式"
            )

        print("=== 无人机配置参数 ===")
        for key, value in uav_config.items():
            print(f"  {key}: {value}")

        print("=== 场景参数 ===")
        print(
            f"无人机初始位置: 纬度 {request.uav_pos.lat}, 经度 {request.uav_pos.lon}, 高度 {request.uav_pos.alt} 米"
        )
        print(
            f"目的地位置: 纬度 {request.destination_pos.lat}, 经度 {request.destination_pos.lon}, 高度 {request.destination_pos.alt} 米"
        )
        print(f"起始时间: {request.current_time}")

        # 执行无人机扫描目标计算
        planner_result = uav_tracking_shooting(
            request.uav_pos.lat,
            request.uav_pos.lon,
            request.uav_pos.alt,
            request.destination_pos.lat,
            request.destination_pos.lon,
            request.destination_pos.alt,
            current_time_obj,
            request.scan_mode
        )

        print("=== 路径规划完成 ===")
        # pprint(f"无人机路径规划结果:\n {planner_result}")

        broadcast_payload = {
            "message": "✅ 无人机路径规划结果计算完成!",
            "message_id": message_id,
            "type": "uavRoute",
            "timestamp": datetime.now().isoformat(),
            "data_count": len(planner_result.get('waypoints')),
            "data": planner_result
        }

        #tag
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        error_msg = f"无人机路线规划过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/broadcast_uavPoint", operation_id="broadcast_uavPoint")
async def broadcast_uavPoint(request: BroadcastUAVPointRequest) -> JSONResponse:
    """
    接收无人机路径规划数据，按每隔{BROADCAST_INTERVAL}秒的时间间隔通过Socket.IO广播waypoints数据到前端。

    Args:
        request: 无人机路径规划数据对象，包含waypoints、searchstart、photopoints等信息

    Returns:
        JSON响应包含处理状态和消息
    """

    BROADCAST_INTERVAL = 1  # 无人机数据广播时间间隔，单位：秒 （） 
    EVENT_TYPE = "uavPoint_update"  # 无人机定位点数据

    POINT_INTERVAL = 3  # 无人机点个数间隔（每隔 POINT_INTERVAL 个点广播一次）

    # 解析请求数据
    

    try:
        # 生成唯一的消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 解析请求数据
        uav_route_data = request.uav_route_data or {}
        num_points = request.num_points or 200 # 无人机广播点个数
        location_name = request.location_name

        if uav_route_data:
            uav_data = uav_route_data.get("data", uav_route_data)
        if uav_route_data:
            uav_data = uav_route_data.get("data", uav_route_data)
            waypoints = uav_data.get("waypoints", [])
        else:
            # 默认读取路径
            uav_route_file = global_config["uav_route_file"]
            # 读取JSON文件
            with open(uav_route_file, "r", encoding="utf-8") as f:
                uav_data = json.load(f)

            # 获取waypoints数据
            waypoints = uav_data.get("waypoints", [])

        if not waypoints:
            raise HTTPException(status_code=400, detail="waypoints数据不能为空")

        # step1: 广播无人机航点定位信息数据
        start_index = uav_data.get("searchstart", [{}])[0].get("Index", 0) # 无人机开始拍摄点位 index

        print(f"📡 无人机路径数据: {len(waypoints)}个路径点")
        print(f"🎯 数据回传间隔: 0.5秒")

        # 设置广播点位参数
        # 无人机开始拍摄前 前N个点位开始
        before_point_index = 0 if start_index - int(num_points / 2 ) < 0 else start_index - int(num_points / 2 )

        # 无人机开始拍位置点位 后显示N个点位结束
        if start_index + int(num_points / 2 ) > len(waypoints):
            if int(num_points / 2 / POINT_INTERVAL) < 10:
                after_point_index = len(waypoints)
            else:
                after_point_index = start_index + int(num_points / 2 )

        if start_index + int(num_points / 2 ) < len(waypoints):
            if int(num_points / 2 / POINT_INTERVAL) < 10:
                after_point_index = start_index + int(num_points / 2 )
            else:
                after_point_index = start_index + 10*POINT_INTERVAL
        
        # 异步广播waypoints数据，每隔0.5秒发送一个点
        async def broadcast_waypoints(start_index=start_index):
            """异步广播waypoints数据"""
            broadcast_data_count = 0
            for index, waypoint in enumerate(waypoints):
                # 从开始拍摄前10个点开始广播，开始后50个点结束（展示效果，节省时间），
                # 真实情况可注释掉这个判断

                
                if index > before_point_index  and index < start_index:
                    if index % 10 == 0:
                        try:
                            # 构建单个waypoint的广播数据
                            waypoint_payload = {
                                "message": "无人机飞行位置点位信息",
                                "message_id": f"{message_id}_{index}",
                                "timestamp": datetime.now().isoformat(),
                                "type":"uavPoint",
                                "event_type": EVENT_TYPE,
                                "data_count": f'{index} / {len(waypoints)}',
                                "data": waypoint,
                                
                            }

                            # 广播当前waypoint
                            await sio.emit(EVENT_TYPE, waypoint_payload)
                            print(f"📍 已回传第 {index + 1}/{len(waypoints)} 个路径点 ")
                            # logger.info(f"📍 已回传第 {index + 1}/{len(waypoints)} 个路径点 航点位置信息: {waypoint}")

                            # 每次广播后都等待指定时间（除了最后一个点）
                            if index < len(waypoints) - 1:
                                await asyncio.sleep(BROADCAST_INTERVAL)
                            broadcast_data_count += 1

                        except Exception as e:
                            print(f"❌ 回传第 {index + 1} 个路径点时发生错误: {str(e)}")
                            continue

                if index > start_index and index < after_point_index:
                    if index % POINT_INTERVAL == 0:
                        try:
                            # 构建单个waypoint的广播数据
                            waypoint_payload = {
                                "message": "无人机飞行位置点位信息",
                                "message_id": f"{message_id}_{index}",
                                "timestamp": datetime.now().isoformat(),
                                "type":"uavPoint",
                                "event_type": EVENT_TYPE,
                                "data_count": f'{index} / {len(waypoints)}',
                                "data": waypoint,
                                
                            }

                            # 广播当前waypoint
                            await sio.emit(EVENT_TYPE, waypoint_payload)
                            print(f"📍 已回传第 {index + 1}/{len(waypoints)} 个路径点 ")
                            # logger.info(f"📍 已回传第 {index + 1}/{len(waypoints)} 个路径点 航点位置信息: {waypoint}")

                            # 每次广播后都等待指定时间（除了最后一个点）
                            if index < len(waypoints) - 1:
                                await asyncio.sleep(BROADCAST_INTERVAL)
                            broadcast_data_count += 1

                        except Exception as e:
                            print(f"❌ 回传第 {index + 1} 个路径点时发生错误: {str(e)}")
                            continue
    
        # 等待异步广播任务完成后再返回响应
        await broadcast_waypoints()
        print(f"✅ 无人机航点位置信息数据广播已全部完成!")
        
        # step2: 广播无人机图像
        uav_image_path = None
        if location_name:
            try:
                broadcast_uavImage_result = await broadcast_uavImage(request)
                # print(broadcast_uavImage_result)
                response_data = json.loads(broadcast_uavImage_result.body.decode())
                
                # print(f"分割\n{response_data}")
                uav_image_path = response_data["data"][0]["filename"]

                print(f"✅ 无人机图像广播成功: {uav_image_path}")
            except Exception as e:
                print(f"⚠️ 广播无人机图像失败: {str(e)}")

        response_data["message"] = "✅ 无人机航点广播成功! -> " + response_data["message"] 

        return JSONResponse(
            content=response_data
        )

    except Exception as e:
        error_msg = f"回传无人机航点位置信息时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.format_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/broadcast_uavImage", operation_id="broadcast_uavImage")
async def broadcast_uavImage(request: UAVImageRequest) -> JSONResponse:
    """
    根据地点名称查找并广播无人机拍摄的最新图像

    Args:
        location_name: 无人机位置，例如"北京"、"北京东单附近"等
            - location: 无人机位置，例如"北京"、"北京东单附近"等
    
    Returns:
        JSONResponse: 包含拍摄的图像路径
            - message: 消息
            - message_id: 消息ID
            - timestamp: 时间戳
            - type: 数据类型
            - data: 图像数据
    """
    
    try:
        # 图像存储目录
        uav_image_dir = Path(global_config["uav_image_dir"])
        
        # 查找包含地点名称的所有图像文件
        matching_files = []
        for img_file in uav_image_dir.glob("*.jpg"):
            if request.location_name in img_file.stem:  # stem 是不含扩展名的文件名
                matching_files.append(img_file)
        
        # 如果没有找到匹配的图像
        if not matching_files:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"未找到包含'{request.location_name}'的图像文件",
                    "image_path": None
                }
            )
        
        # 按文件名中的时间信息排序,选择最新的
        # 文件名格式: "地点_年月日-时分秒.jpg" (如 "北京_20250103-113020.jpg")
        # def extract_timestamp(file_path: Path) -> str:
        #     """从文件名中提取时间戳字符串用于排序"""
        #     try:
        #         # 获取文件名(不含扩展名): "北京_20250103-113020"
        #         filename = file_path.stem
        #         # 分割获取时间部分: "20250103-113020"
        #         time_part = filename.split("_")[-1]
        #         return time_part
        #     except:
        #         return "00000000-000000"  # 解析失败返回最小时间
        
        # # 按时间戳降序排序,第一个就是最新的
        # latest_image = sorted(matching_files, key=extract_timestamp, reverse=True)[0]
        
        # print(f"✅ 找到 {len(matching_files)} 个匹配'{location_name}'的图像")
        # print(f"📸 选择最新图像: {latest_image.name}")

        # 复制图像到目标目录
        uav_image = PHOTOGRAPHS_DIR / matching_files[0].name
        shutil.copy(matching_files[0], uav_image)

        broadcast_payload = {
            "message": "🛜 无人机拍摄图像广播成功!",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "type": "uavImage",
            "base_url": PHOTOGRAPHS_URL,
            "extension": {
                "description": f"无人机已成功返回 {request.location_name} 的航拍图像!",
            },
            "data_count": 1,
            "data": [{
                "location":f"[{request.location_longitude}, {request.location_latitude}]",
                "time":datetime.now().isoformat(),
                "filepath": str(uav_image),
                "filename": uav_image.name,
            }],
            "event_type": "uavImage_update",
        }

        # 无人机图像广播
        await sio.emit("uavImage_update", broadcast_payload)
        print(f"✅ 无人机图像广播成功: {uav_image.name}")

        return JSONResponse(content=broadcast_payload)
        
    except Exception as e:
        error_msg = f"广播无人机图像时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/img_predictor", operation_id="img_predictor")
async def img_predictor(request: PredictRequest):
    """
    图片目标检测API

    Args:
        request: 包含图像文件名的请求对象
            - image_path: 待预测的图像文件路径(支持 .jpg, .jpeg, .png, .tif) (必选)

    Returns:
        JSON响应：检测结果和结果图片URL列表
    """

    try:
        message_id = str(uuid.uuid4())
        # 检查图像文件是否存在
        image_path, image_url = resolve_file_path(request.image_path)

        # 调用YOLO预测函数
        predict_result = predict(str(image_path), str(RESULTS_DIR))

        data_count = predict_result.get("objects_counts")
        data_description = ""
        for key, value in data_count.items():
            if key == 'total':
                data_description += f" 总数{value}辆; "
            else:
                data_description += f"检测到{key} {value} 辆; "
    
        # 数据广播payload
        broadcast_payload = {
            "message": "✅ 图像预测成功!",
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "type": "imgPredict",
            "base_url": PREDICT_URL,
            "extension": {
                "description": f"共检测到车辆 {data_count["total"]} 辆",
            },
            "data_count": data_count,
            "data": [predict_result]
        }

        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        error_msg = f"图像预测过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/img_cropper", operation_id="img_cropper")
async def img_cropper(request: CropRequest):
    """
    图像目标裁剪API

    Args:
        request: 包含图像文件名的请求对象
            - predicted_json_path: 预测结果JSON文件名 (支持 .jpg, .jpeg, .png, .tif)。
                例如: "group3-1(12).jpg" -> 自动找到对应的 "group3-1(12)_predict_info.json"

    Returns:
        JSON响应:裁剪后图像的URL列表。
    """

    try:
        message_id = str(uuid.uuid4())
        # 根据图像文件名构造对应的检测结果JSON文件路径
        json_path, image_url = resolve_file_path(request.predicted_json_path)

        # 调用裁剪函数
        cropped_results = crop_objects(input_label_path=json_path) # 返回裁剪后的图像路径列表
        

        print(f"成功裁剪 {len(cropped_results)} 个目标图像")
        pprint(f"{cropped_results}\n")

        with open(request.predicted_json_path, "r", encoding="utf-8") as f:
            predicted_json_data = json.load(f)
            detections = predicted_json_data.get("detection", [])
            # pprint(f"{detections}")

            # 4. 将绝对路径转换为可访问的URL
            file_urls = []
            if cropped_results:
                for file_path,detection in zip(cropped_results,detections):
                    # 获取文件名
                    filename = Path(file_path).name
                    # 构造可访问的URL
                    file_url = f"results/objects/{filename}"
                    detection["filename"] = filename
                    file_urls.append(file_url)

        # 数据广播payload
        broadcast_payload = {
            "message": "✂️ 图像裁剪成功完成!",
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "type": "imgCrop",
            "base_url": CROP_URL,
            "extension": {
                "description": f"成功提取 {len(cropped_results)} 个目标图像"
            },
            "objects_list": file_urls,
            "data_count": len(cropped_results),
            "data": detections
        }

        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)
        # 6. 返回成功响应
        return broadcast_result

    except Exception as e:
        error_msg = f"图像裁剪过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post(
    "/objects_searcher",
    operation_id="objects_searcher",
)
async def objects_searcher(request: TargetSearchRequest):
    """
    根据目标图像搜索相似图像

    Args:
        request: 包含目标图像路径列表的请求对象
            - objects_list: 车辆目标图像路径列表 (必选)
            - id: 图像id列表，若为空，则不进行id筛选
            - time_start: 开始时间，若为空，则不进行时间筛选
            - time_end: 结束时间，若为空，则不进行时间筛选
            - extent: 搜索范围，若为空，则不进行范围筛选
        注意：相似目标图像搜索时，只启用 objects_list参数

    Returns:
        JSON响应:目标图像检索结果和结果图片地址

    """
    try:
        message_id = str(uuid.uuid4())

        if not request.objects_list:
            raise HTTPException(status_code=400, detail="objects_list cannot be empty")

        all_results = []
        processed_count = 0
        errors = []
        all_results_filepaths = []
        all_descriptions = []

        # 处理每张图像
        for i, input_image in enumerate(request.objects_list):
            # 使用通用函数解析图像路径
            image_path, error_msg = resolve_file_path(input_image)
            
            if error_msg:
                # 解析失败,记录错误并跳过
                errors.append(error_msg)
                continue
            print(f"\n{'='*100}")
            print(f"处理第 {i + 1} 张图像: {os.path.basename(image_path)}")
            print(f"{'='*100}")

            try:
                # 检索目标图像（直接传递文件路径）
                # target_results = search_image_from_milvus(query_image=image_path, query_type="target")
                target_results = clip_searcher(query_image=image_path,configs=global_config)

                print(f"找到 {len(target_results)} 个目标图像相似结果")

                # 添加额外字段
                    # - "target"字段,表明查询目标对应的切分目标序号
                    # - "filename"字段,表明查询目标对应的文件名
                for item in target_results:
                    print(item)
                    item["filename"] = os.path.basename(item.get("target_image_path", "N/A"))
                    item["target"] = i + 1

                # 保存单个图像的目标图像检索结果到objects_search目录
                image_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
                target_results_filename = f"{image_name_without_ext}_target.json"
                target_results_filepath = os.path.join(
                    OBJECTS_SEARCH_DIR, target_results_filename
                )

                # 确保目录存在
                os.makedirs(OBJECTS_SEARCH_DIR, exist_ok=True)

                with open(target_results_filepath, "w", encoding="utf-8") as f:
                    json.dump(target_results, f, ensure_ascii=False, indent=2)
                print(f"已保存目标结果到: {target_results_filepath}")

                # 分析标签
                label_analysis = analyze_target_label(target_results, top_n=10)
                print(f"标签分析结论: {label_analysis.get('conclusion', 'N/A')}")

                all_descriptions.append(f"第{i+1}个目标(共{len(request.objects_list)}个目标)："+label_analysis.get("conclusion", "N/A")) # 标签分析结论
                all_results.append(target_results)                               # 目标图像检索结果
                all_results_filepaths.append(target_results_filepath)
                processed_count += 1
                print(f"成功处理图像 {i + 1}/{len(request.objects_list)}")

            except Exception as e:
                error_msg = f"处理图像 {image_path} 时出错: {str(e)}"
                print(f"错误详情: {traceback.format_exc()}")
                errors.append(error_msg)

        broadcast_payload = {
            "message":"✅ 相似目标检索成功!",
            "message_id": message_id,
            "timestamp":datetime.now().isoformat(),
            "type":"imgObject",
            "base_url": OBJECTS_URL,
            "extension":{
                "description": all_descriptions,
                "filepath_list": all_results_filepaths
                },
                "data_count":[len(result) for result in all_results],
                "data":all_results
            }
        
        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        error_msg = f"相似目标检索过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    

@app.post("/history_searcher",operation_id="history_searcher")
async def history_searcher(request: TargetSearchRequest):
    """
    处理图像并进行历史图像搜索

    Args:
        request: 包含图像文件路径列表的请求对象
            - objects_list: 车辆目标图像文件路径列表 (必选)。
            - id: 图像id列表，若为空，则不进行id筛选
            - time_start: 开始时间，若为空，则不进行时间筛选
            - time_end: 结束时间，若为空，则不进行时间筛选
            - extent: 搜索范围，若为空，则不进行范围筛选

    Returns:
        JSON响应：历史图像检索结果和结果图片地址
    """

    try:
        message_id = str(uuid.uuid4())

        if not request.objects_list:
            raise HTTPException(status_code=400, detail="objects_list cannot be empty")
            raise HTTPException(
                status_code=400, detail="input objects_list cannot be empty"
            )

        processed_count = 0
        errors = []
        all_results = []
        all_results_filepaths = []
        all_descriptions = []

        # 处理每张图像
        for i, image_item in enumerate(request.objects_list):
            # 判断是否是URL

            print(f"\n{'='*60}")
            # 使用通用函数解析图像路径
            image_path, error_msg = resolve_file_path(image_item)
            
            if error_msg:
                # 解析失败,记录错误并跳过
                print(f"警告: {error_msg}")
                errors.append(error_msg)
                continue

            print(f"处理第 {i + 1} 张图像: {os.path.basename(image_path)}")

            try:
            
                # ----------  stpe1：从 Milvus 获取相似度信息 ----------

                # milvus_results = search_image_from_milvus(query_image=image_path, query_type="history")
                milvus_results = clip_searcher(query_image=image_path,configs=global_config)
                print(f"Milvus历史图像相似结果： {len(milvus_results)} 个")
                # pprint(milvus_results)
                print("-"*100)

                # ----------  step2：根据milvus检索结果id参数和输入的时间地点完成数据库检索 ----------

                result_ids = []
                for result in milvus_results:
                    result_ids.append(result["id"])

                # pprint(f"milvus history ids:\n: {result_ids}")

                table_name = global_config["mysql_image"]['table_name'] 

                mysql_results = query_image_data(
                    id=result_ids,
                    time_start=request.time_start,
                    time_end=request.time_end,
                    extent=request.extent,
                    table_name=table_name,
                )
                print(f"MySQL历史图像检索结果： {len(mysql_results)} 个")

                # 合并 Milvus 的相似度信息和 MySQL 的元数据
                # 创建 id 到 distance 的映射
                id_to_distance = {item["id"]: item["distance"] for item in milvus_results}
                
                # 为 MySQL 结果添加 distance 字段
                history_results = []
                for mysql_item in mysql_results:
                    item_id = mysql_item.get("id")
                    if item_id in id_to_distance:
                        # 合并数据：MySQL 元数据 + Milvus 相似度
                        merged_item = {
                            "id": item_id,
                            "distance": id_to_distance[item_id],  # 来自 Milvus
                            **mysql_item,  # 来自 MySQL 的所有字段
                            "filename": os.path.basename(mysql_item.get("target_image_path", "N/A"))
                        }
                        history_results.append(merged_item)
                    else:
                        # 如果在 Milvus 结果中找不到对应的 distance，使用默认值
                        print(f"⚠️ 警告：ID {item_id} 在 Milvus 结果中未找到，使用默认 distance")
                        mysql_item["distance"] = 0.0
                        history_results.append(mysql_item)

                # 按distance降序排序（distance越大相似度越高）
                history_results = sorted(history_results, key=lambda x: x.get("distance", 0), reverse=True)
                print(f"✅ 历史图像结果已按相似度排序，共 {len(history_results)} 条")

                # 添加"target"字段,表明查询目标对应的切分目标序号
                for item in history_results:
                    item["target"] = i + 1

                # 设置保存历史图像检索结果到history_search目录
                image_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
                history_results_filename = f"{image_name_without_ext}_history.json"
                history_results_filepath = os.path.join(
                    HISTORY_SEARCH_DIR, history_results_filename
                )

                # 保存检索结果
                with open(history_results_filepath, "w", encoding="utf-8") as f:
                    json.dump(history_results, f, ensure_ascii=False, indent=2)

                # 分析标签
                label_analysis = analyze_target_label(history_results, top_n=10)
                print(f"标签分析结论: {label_analysis.get('conclusion', 'N/A')}")

                # all_results.append(image_response)
                all_results.append(history_results)
                all_results_filepaths.append(history_results_filepath)
                all_descriptions.append(label_analysis.get("conclusion", "N/A"))
                processed_count += 1
                print(f"成功处理图像 {i + 1}/{len(request.objects_list)}")
                print(f"{'='*100}")

            except Exception as e:
                error_msg = f"处理图像 {image_path} 时出错: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                # 可以选择继续处理其他图像或抛出异常

        # 数据广播payload
        broadcast_payload = {    
                "message":"✅ 历史目标检索成功!",
                "message_id":message_id,
                "timestamp":datetime.now().isoformat(),
                "type":"imgHistory",
                "base_url":HISTORY_URL,
                "extension":{
                    "description": all_descriptions,
                    "filepath_list": all_results_filepaths
                },
                "data_count":[len(result) for result in all_results],
                "data":all_results
        }
        
        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result
    except Exception as e:
        error_msg = f"历史目标检索过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/analyze_router",operation_id="analyze_router",)
async def analyze_router(request: RouteAnalysisRequest):
    """
    分析车辆历史图像搜索结果的路径信息

    Args:
        request: 包含历史图像搜索结果的JSON文件路径列表的请求对象
            - json_list: 历史图像搜索结果JSON文件路径列表

    Returns:
        response: 车辆历史图像搜索结果的路径信息
    """

    try:
        message_id = str(uuid.uuid4())
        all_results = analyze_route_main(request.history_json_list)

        # 构建路线描述字符串
        # all_results 是嵌套列表: [[routes_from_file1], [routes_from_file2], ...]
        route_description = ""
        total_routes = 0
        
        if all_results and any(all_results):  # 检查是否有非空结果
            route_descriptions = []  # 使用列表收集每条路线描述
            route_counter = 1
            
            # 遍历每个JSON文件的分析结果
            for file_idx, routes_list in enumerate(all_results, 1):
                if routes_list:  # 如果该文件有路线结果
                    # 遍历该文件的所有路线
                    for route in routes_list:
                        # 构建紧凑的单行描述
                        parts = [
                            f"轨迹{route_counter}({route['route_id']})",
                            f"起点({route['start_latitude']:.6f},{route['start_longitude']:.6f})",
                            f"终点({route['end_latitude']:.6f},{route['end_longitude']:.6f})",
                            f"时间({route['start_timestamp']}→{route['end_timestamp']})",
                        ]
                        
                        # 添加可选信息
                        if 'total_distance_km' in route:
                            parts.append(f"距离{route['total_distance_km']:.2f}km")
                            parts.append(f"速度{route['average_speed_kmh']:.2f}km/h。 ")
                        
                        # parts.append(f"点数{route['num_points']}。 ")
                        
                        # 用分号连接各部分
                        route_descriptions.append("; ".join(parts))
                        route_counter += 1
                        total_routes += 1
            
            # 用换行符连接所有路线,方便阅读
            route_description = "可能的车辆行驶路线:\n" + "\n".join(route_descriptions)
        else:
            route_description = "未发现可能的行驶路线"

        # 打印到控制台（可选）
        print(route_description)

        # 处理分析结果并生成响应
        broadcast_payload = {
            "message":"✅ 车辆路径分析成功!",
            "message_id":message_id,
            "timestamp":datetime.now().isoformat(),
            "type":"vehicleRoute", 
                "filepath_list":request.history_json_list,
                "extension":{
                    "description": route_description,  # 添加路线描述字符串
                    "route_count": total_routes,  # 总路线数
                },
                "data_count":[len(result) for result in all_results],
                "data":all_results
            
        }
        
        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        error_msg = f"车辆路径分析过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


# =====================================================   通用工具路由  =====================================================

#tag 广播路由接口


@app.post("/broadcast_default", operation_id="broadcast_default")
async def broadcast_default(input_data: Dict[str, Any]) -> JSONResponse:
    """
    接收装备数据，利用Socket.IO将获取的数据广播到前端。

    Args:
        input_data: Dict[str, Any] 输入数据对象，包含各种信息

    Returns:
        JSON响应包含处理状态和消息
    """

    if input_data.get("event_type", "broadcast_update"):
        EVENT_TYPE = input_data.get("type", "broadcast_update") + "_update" # 输入数据事件类型
    else:
        EVENT_TYPE = "broadcast"  # 输入数据事件类型

    try:
        # 生成唯一的消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # 构建广播数据（将Pydantic对象转换为字典）
        input_data["message"] = input_data["message"] + " --> 🛜 数据广播成功!"
        input_data["message_id"] = message_id
        input_data["timestamp"] = timestamp
        input_data["event_type"] = EVENT_TYPE
      
        # 广播数据到所有连接的客户端
        await sio.emit(EVENT_TYPE, input_data)
        if input_data.get("type"):
            print(f"🛜 广播数据类型: {input_data['type']}")
        else:
            print(f"⚠️ 广播输入数据 No [ Type ] field!")

        if input_data.get("data_count"):
            print(f"🛜 广播数据数量: {input_data['data_count']}条")
        else:
            print(f"⚠️ 广播输入数据 No [ data_count ] field!")

        return JSONResponse(content=input_data)

    except Exception as e:
        error_msg = f"广播装备数据时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/broadcast_equipData", operation_id="broadcast_equipData")
async def broadcast_equipData(equip_data: Dict[str, Any] | List[Dict[str, Any]] = None) -> JSONResponse:
    """
    接收装备数据，利用Socket.IO将获取的数据广播到前端。

    Args:
        equip_data: 装备数据对象，包含装备的各种信息

    Returns:
        JSON响应包含处理状态和消息
    """

    EVENT_TYPE = "equipData_update"  # 装备数据事件类型

    try:
        # 生成唯一的消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        if isinstance(equip_data, dict):
            equip_data = equip_data["data"]
        elif isinstance(equip_data, list):
            pass
        else:
            raise HTTPException(status_code=400, detail="equip_data must be a dictionary or a list of dictionaries")

        # 构建广播数据（将Pydantic对象转换为字典）
        broadcast_payload = {
            "message": "🛜 装备数据广播成功!",
            "message_id": message_id,
            "timestamp": timestamp,
            "type": "equipData",
            "event_type": EVENT_TYPE,
            "data_count": len(equip_data),
            "data": equip_data,
        }
      
        # 广播数据到所有连接的客户端
        await sio.emit(EVENT_TYPE, broadcast_payload)
        print(f"📡 广播装备数据: {len(equip_data)}条")

        return JSONResponse(content=broadcast_payload)

    except Exception as e:
        error_msg = f"广播装备数据时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/broadcast_geocode", operation_id="broadcast_geocode")
async def broadcast_geocode(request: GeocodeRequest):
    """
    接受地理位置查询请求，通过WebSocket向所有连接的客户端广播地理位置查询指令

    Args:
        request: 地理位置查询请求体
            - 位置: 地理位置（城市名、地区名）
            - 缩放级别: 缩放级别（0-18）

    Returns:
        JSON响应包含地理位置查询结果
        - 消息: 地理位置查询结果
        - 指令: 地理位置查询指令
        - 描述: 地理位置查询完成，已在地图上添加定位标记
    """

    EVENT_TYPE = "geocode_update"

    try:
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # 构建地理位置查询指令
        geocoder_payload = {
            "message": "地理位置查询已启动",
            "message_id": message_id,
            "timestamp": timestamp,
            "type": "geocode",
            "event_type": EVENT_TYPE,
            "data": request.model_dump()  # 转换为字典以支持 JSON 序列化
        }

        # 通过WebSocket向所有连接的客户端广播地理位置查询指令
        await sio.emit(EVENT_TYPE, geocoder_payload)
        print(f"Geocoder command broadcasted to clients: {json.dumps(geocoder_payload)}")

        return JSONResponse(content=geocoder_payload)

    except Exception as e:
        error_msg = f"地理位置查询时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/socketio/rooms", operation_id="get_socketio_rooms")
async def get_socketio_rooms():
    """
    获取当前所有Socket.IO房间信息

    Returns:
        JSON响应包含房间列表和客户端数量
    """

    try:
        rooms_info = {}
        namespace = "/"

        # 获取所有房间信息
        if hasattr(sio.manager, "rooms") and namespace in sio.manager.rooms:
            for room_name, clients in sio.manager.rooms[namespace].items():
                rooms_info[room_name] = {
                    "name": room_name,
                    "client_count": len(clients),
                    "clients": list(clients),
                }

        return JSONResponse(
            content={
                "success": True,
                "namespace": namespace,
                "total_rooms": len(rooms_info),
                "rooms": rooms_info,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        error_msg = f"获取房间信息时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


# ==========================================================  通用工具路由接口  ==========================================================

#tag 通用工具路由接口

@app.post("/img_matcher", operation_id="img_matcher")
async def img_matcher(request: FileListRequest) -> Dict[str, Any]:
    """
    根据文件名智能匹配并获取相关的目标识别、影像裁剪、目标检索和历史检索等影像文件路径。

    Args:
        request: 图片文件名请求对象
            - img_name: 照片文件名（如 'TK08_11.jpg'）

    Returns:
        JSON响应：包含 photographs, predict, objects, history_search, objects_search 的字典
        例如：
        {
            "photographs": "TK08_11.jpg",
            "predict": "TK08_11_predict_info.json",
            "objects": ["TK08_11_armored_vehicle_1.jpg", "TK08_11_tank_1.jpg", ...],
            "history_search": ["TK08_11_armored_vehicle_1_history.json","TK08_11_tank_1_history.json",...],
            "objects_search": ["TK08_11_armored_vehicle_1_label.json","TK08_11_tank_1_label.json",...]
        }
    """

    ROOT_DIR = "results"

    try:
        message_id = str(uuid.uuid4())
        # 从文件名中提取基础名称（去掉扩展名）
        filename = request.img_name  # 获取文件名
        base_name = Path(filename).stem  # image1.png -> image1
        print(f"正在为文件 '{filename}' 查找相关数据，基础名称: '{base_name}'")

        # 初始化结果字典
        result = {
            "photographs": None,
            "predict": None,
            "objects": [],
            "history_search": [],
            "objects_search": [],
        }

        # 1. 查找 photographs 目录中的匹配文件（应该等于输入的filename）
        photographs_path = PHOTOGRAPHS_DIR / filename
        if photographs_path.exists():
            result["photographs"] = ROOT_DIR + "/photographs/" + filename
        else:
            print(f"未找到照片文件: {filename}")

        # 2. 查找 predicts 目录中的匹配文件（格式：basename_predict_info.json）
        predict_name = f"{base_name}_predict.json"
        predict_path = PREDICTS_DIR / predict_name
        if predict_path.exists():
            result["predict"] = ROOT_DIR + "/predicts/" + predict_name
        else:
            print(f"未找到预测文件: { predict_path}")

        # 3. 查找 objects 目录中的匹配文件（格式：basename_*.jpg）
        objects_files = []
        for obj_file in OBJECTS_DIR.glob(f"{base_name}_*.jpg"):
            objects_files.append(obj_file.name)

        if objects_files:
            result["objects"] = sorted(
                [ROOT_DIR + "/objects/" + obj_file for obj_file in objects_files]
            )
        else:
            print(f"未找到目标文件，查找模式: {base_name}_*.jpg")

        # 4. 查找 history_search 目录中的匹配文件（格式：basename*_history.json）
        history_files = []
        for history_file in HISTORY_SEARCH_DIR.glob(f"{base_name}*_history.json"):
            history_files.append(history_file.name)

        if history_files:
            result["history_search"] = sorted(
                [
                    ROOT_DIR + "/history_search/" + history_file
                    for history_file in history_files
                ]
            )
        else:
            print(f"未找到历史搜索文件，查找模式: {base_name}*_history.json")

        # 5. 查找 objects_search 目录中的匹配文件（格式：basename*_target.json）
        target_files = []
        for target_file in OBJECTS_SEARCH_DIR.glob(f"{base_name}*_target.json"):
            target_files.append(target_file.name)

        if target_files:
            result["objects_search"] = sorted(
                [
                    ROOT_DIR + "/objects_search/" + target_file
                    for target_file in target_files
                ]
            )
        else:
            print(f"未找到目标搜索文件，查找模式: {base_name}*_target.json")

        pprint(f"匹配结果汇总:\n {result}")

        # 数据广播payload
        broadcast_payload = {
            "message": "✅ 图片匹配结果计算完成!",
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "type": "fileList",
            "base_url": BASE_URL,
            "data_count": len(result),
            "data": result
        }

        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        error_msg = f"匹配文件 {filename} 时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/equipment_query", operation_id="equipment_query")
async def equipment_query(request: EquipmentQueryRequest):
    """
    数据库装备数据查询功能：接受装备数据查询请求，通过 socketIO 向所有连接的客户端广播装备查询结果。

    Args:
        request: 装备数据查询请求对象
            - extent: 地理范围：[minX, minY, maxX, maxY]，例如，[116.397428, 39.90923, 116.405428, 39.91723]
            - keyRegion: 关键区域：例如，北京
            - topic: 专题：例如，太空态势专题
            - layer: 装备空间分布层级：例如，space、air、ground、sea等
            - camp: 阵营：例如，红方、蓝方
            - status: 状态：例如，可用、 不可用
            - database: 数据库：例如，equipment_db
            - table_name: 表名称：例如，equipment_data

    Returns:
        JSON响应：{
            "message": "装备数据查询完成！",
            "message_id": message_id,
            "timestamp": timestamp,
            "type": "equipData",
            "data_count": len(equipment_data),
            "data": equipment_data,
        }
    """

    EVENT_TYPE = "equipment_query"

    try:
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # 查询装备数据
        equipment_data = query_equipment_data(
            extent=request.extent, 
            keyRegion=request.keyRegion, 
            topic=request.topic, 
            layer=request.layer, 
            camp=request.camp, 
            status=request.status,
            database=request.database,
            table_name=request. table_name
        )

        # 数据广播payload
        broadcast_payload = {
            "message": "装备数据查询完成！",
            "message_id": message_id,
            "timestamp": timestamp,
            "type": "equipData",
            "data_count": len(equipment_data),
            "data": equipment_data,
        }

        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except Exception as e:
        error_msg = f"装备数据查询时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/RSimage_query", operation_id="RSimage_query")
async def RSimage_query(request: RSimageRrequest):
    """
    从本地MySQL数据库查询获取遥感影像元数据。

    Args:
        request: RS_params请求对象，包含数据库连接参数和查询条件：

            # 数据查询参数
            - acquisitionTime: List[Dict[str, int]] = None # 影像采集时间 {"Start": start_timestamp,"End": end_timestamp}
            - extent: List[float] = None  # bbox[minX(西), minY(北), maxX(东), maxY(南)]
            - cloud_percent_min: Optional[int] = 0  # 云量最小值
            - cloud_percent_max: Optional[int] = 20  # 云量最大值
            - limit: Optional[int] = None  # 限制返回记录数

            # 数据库连接参数
            - host: 数据库地址
            - port: 数据库端口
            - user: 数据库用户名
            - password: 数据库密码
            - database: 数据库名称，默认RS_images_db
            - table_name: 表名称，默认RS_images_metadata

    Returns:
        JSON响应：{
            "message": "遥感影像数据查询成功",
            "message_id": message_id,
            "timestamp": timestamp,
            "event_type": EVENT_TYPE,
            "data": rs_images_data,
        }
    """

    try:
        # 生成唯一的消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        print(f"📡 开始获取遥感影像元数据...")
        
        # 调试：打印接收到的请求参数
        request_data = request.model_dump(exclude_none=False)  # 确保包含 None 值
        print(f"\n{'='*80}")
        print(f"🔍 API接收到的请求参数:")
        print(f"  acquisitionTime: {request_data.get('acquisitionTime')}")
        print(f"  acquisitionTime type: {type(request_data.get('acquisitionTime'))}")
        print(f"  extent: {request_data.get('extent')}")
        print(f"  cloud_percent_min: {request_data.get('cloud_percent_min')}")
        print(f"  cloud_percent_max: {request_data.get('cloud_percent_max')}")
        print(f"  完整参数: {request_data}")
        print(f"{'='*80}\n")

        # Step 1: 调用 get_satellite_from_mysql 获取遥感影像元数据
        satellite_metadata = get_satellite_from_mysql(**request_data)

        if not satellite_metadata or len(satellite_metadata) == 0:
            raise HTTPException(status_code=404, detail="未找到符合条件的遥感影像数据")

        print(f"✓ 从数据库获取到 {len(satellite_metadata)} 条遥感影像记录")

        # 记录广播日志
        broadcast_payload = {
            "message": "遥感影像数据查询成功!",
            "message_id": message_id,
            "timestamp": timestamp,
            "type": "RSimage",
            "base_url": "/data/RS_images/",
            "data_count": len(satellite_metadata),
            "data": satellite_metadata,
        }

        # 数据广播
        broadcast_result = await broadcast_default(broadcast_payload)

        return broadcast_result

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"广播遥感影像数据时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post('/data_search', operation_id="data_search")
async def search_equipment_from_sql(request: str = Body(...)) -> JSONResponse:
    """ 
    基于自然语言查询数据库的Text2SQL服务。

    触发关键词: 查询、列表、有哪些专题
    
    Args:
        query: 用户的查询问题，默认查询所有可用专题
    
    Returns:
        查询结果

    """
    equip_config = load_config()["mysql_equipment"]

    try:
        # 检查Text2SQL组件是否初始化成功
        text2sql_chain = init_text2sql(
            database=equip_config["database"],  # 默认数据库名称
            host=equip_config["host"],
            user=equip_config["user"],
            password=equip_config["password"],
            port=equip_config["port"],
        )

        if text2sql_chain is None:
            return JSONResponse(content={
                'error': 'Text2SQL功能未初始化，请检查配置',
                'success': False
            }), 500

        # 获取请求参数
        
        # 使用Text2SQL链进行查询
        result = text2sql_chain.invoke(request)
        # print("******************:", result["intermediate_steps"][3])
        # print(type(result["intermediate_steps"][3]))
        
        # 提取结果信息
        table_info = result["intermediate_steps"][0] if result.get("intermediate_steps") else None
        sql_query = result["intermediate_steps"][1] if result.get("intermediate_steps") and len(result["intermediate_steps"]) > 1 else None
        # query_result = result.get("result", "")
        query_result = result["intermediate_steps"][3] if result.get("intermediate_steps") and len(result["intermediate_steps"]) > 2 else None     

        # 获取数据表的字段信息（可选，根据需要启用）, 从 SQL 查询中提取表名
        field_info = None
        if sql_query:
            # 简单提取 FROM 后面的表名
            import re
            match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
            if match:
                table_name = match.group(1)
                field_info = get_field_names_only(
                    table_name=table_name,
                    host=equip_config["host"],
                    user=equip_config["user"],
                    password=equip_config["password"],
                    port=equip_config["port"],
                    database=equip_config["database"]
                    )

        # 返回结构化结果
       
        # query_result = ast.literal_eval(query_result)

        # 修改方案2：添加类型检查
        if isinstance(query_result, str):
            query_result = ast.literal_eval(query_result)
        elif isinstance(query_result, list):
            # 已经是列表，直接使用
            pass
        else:
            # 其他类型处理
            query_result = []
        
        response_data = {
            'success': True,
            # 'natural_language_query': natural_language_query,
            # 'generated_sql': sql_query,
            # 'sql_query': sql_query,
            # 'table_info': table_info,

            'field': field_info,  # 如需返回字段信息，取消注释
            'data_count': len(query_result),
            'data': query_result, # 将字符串转换为列表
        }

        print("\n===   数据查询完成，结果如下：  ===")
        print(response_data)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        return JSONResponse(content={
            'error': f'查询处理失败: {str(e)}',
            'success': False
        }), 500


@app.post('/satellite_time_search',operation_id="satellite_time_search")
async def satellite_time_search(request: strTypeRequest) -> JSONResponse:

    try:
        if "霍尔木兹海峡" in request.keyRegion:
            print(f"开始查询霍尔木兹海峡地区的卫星时间信息")
            with open(os.path.join(DATA_DIR, "satellite_time_table.txt"), "r", encoding='utf-8') as f:
                satellite_time_table = f.read()

            return JSONResponse(content=json.dumps(satellite_time_table, default=str))
        else:
            return json.dumps({"message": "未找到霍尔木兹相关的卫星时间信息"}, default=str)

    except Exception as e:
        error_msg = f"卫星时间查询过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"错误详情:\n {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


# ============================================    function tools   =================================================

#tag 功能函数

def parse_time_string(time_str: str) -> datetime.time:
    """
    解析时间字符串，支持多种格式

    Args:
        time_str: 时间字符串

    Returns:
        datetime.time 对象

    Raises:
        ValueError: 时间格式不正确
    """

    if not time_str or not isinstance(time_str, str):
        raise ValueError(f"时间字符串不能为空或非字符串类型: {time_str}")

    # 清理输入字符串
    time_str = time_str.strip()

    # 支持的时间格式
    time_formats = [
        "%H:%M:%S",  # 10:30:45
        "%H:%M",  # 10:30 (自动补充秒数为00)
        "%I:%M:%S %p",  # 10:30:45 AM/PM
        "%I:%M %p",  # 10:30 AM/PM
    ]

    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(time_str, fmt).time()
            print(f"✅ 时间解析成功: '{time_str}' -> {parsed_time} (格式: {fmt})")
            return parsed_time
        except ValueError:
            continue

    # 如果所有格式都失败，抛出详细错误
    raise ValueError(
        f"无法解析时间字符串 '{time_str}'。"
        f"支持的格式: HH:MM:SS, HH:MM, HH:MM:SS AM/PM, HH:MM AM/PM"
    )


def resolve_file_path(input_file: str, save_dir: Optional[str] = None) -> tuple[str, str | None]:
    """
    解析图像路径,支持URL和本地路径
    
    功能:
    1. 如果输入是本地服务器URL,直接转换为本地文件路径(避免死锁)
    2. 如果输入是外部URL,下载到本地指定目录
    3. 如果输入是本地路径(相对或绝对),直接使用
    
    Args:
        input_file: 图像输入,可以是URL或本地路径
        save_dir: 下载的图像保存目录名(相对于ROOT_DIR),默认为None
        
    Returns:
        tuple: (本地图像路径, 错误信息)
            - 成功时返回 (local_path, None)
            - 失败时返回 (None, error_message)
    
    Examples:
        >>> # URL输入
        >>> path, err = resolve_image_path("http://localhost:5000/results/uav_way/test.jpg")
        >>> # 相对路径输入
        >>> path, err = resolve_image_path("results/uav_way/test.jpg")
        >>> # 绝对路径输入
        >>> path, err = resolve_image_path("D:/images/test.jpg")
    """
    try:
        # 判断是否是URL
        if input_file.startswith("http://") or input_file.startswith("https://"):
            # 解析URL
            parsed_url = urlparse(input_file)
            is_local_server = parsed_url.hostname in ['localhost', '127.0.0.1'] and parsed_url.port == PORT
            
            if is_local_server:
                # 本地服务器URL,直接转换为本地文件路径(避免HTTP请求死锁)
                # 例如: http://localhost:5000/results/uav_way/xxx.jpg -> ROOT_DIR/results/uav_way/xxx.jpg
                relative_path = parsed_url.path.lstrip('/')
                relative_path = unquote(relative_path)  # 解码URL编码的中文字符
                local_image_path = os.path.join(ROOT_DIR, relative_path)
                
                if os.path.exists(local_image_path):
                    print(f"✅ 识别为本地服务器路径,直接使用: {relative_path}")
                    return local_image_path, None
                else:
                    error_msg = f"本地文件不存在: {local_image_path}"
                    print(f"❌ {error_msg}")
                    return None, error_msg
                    
            else:
                # 外部URL,需要下载
                print(f"🌐 检测到外部URL,开始下载: {input_file}")
                
                # 使用 HEAD 请求检查 URL 是否有效
                response_head = requests.head(input_file, timeout=30)
                
                if response_head.status_code == 200:
                    # 下载图像到本地目录
                    response_get = requests.get(input_file, timeout=60)
                    
                    # 从 URL 中提取文件名并解码中文字符
                    filename = os.path.basename(parsed_url.path)
                    filename = unquote(filename)  # 解码URL编码的中文字符
                    
                    # 保存文件到本地
                    save_path = save_dir or os.path.join(ROOT_DIR, "data/upload_images")
                    os.makedirs(save_path, exist_ok=True)
                    local_image_path = os.path.join(save_path, filename)
                    
                    with open(local_image_path, "wb") as f:
                        f.write(response_get.content)
                    
                    print(f"✅ 成功下载图像: {filename}")
                    return local_image_path, None
                else:
                    error_msg = f"图像URL返回状态码 {response_head.status_code}: {input_file}"
                    print(f"❌ {error_msg}")
                    return None, error_msg
                    
        else:
            # 本地路径(相对或绝对)
            if not os.path.isabs(input_file):
                # 相对路径,转换为绝对路径
                image_path = os.path.join(ROOT_DIR, input_file)
            else:
                # 绝对路径,直接使用
                image_path = input_file
            
            if os.path.exists(image_path):
                print(f"✅ 使用本地路径: {image_path}")
                return image_path, None
            else:
                error_msg = f"图像文件不存在: {image_path}"
                print(f"❌ {error_msg}")
                return None, error_msg
                
    except requests.exceptions.Timeout as e:
        error_msg = f"请求超时,无法访问URL: {input_file} - {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg
        
    except requests.exceptions.RequestException as e:
        error_msg = f"请求失败: {input_file} - {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg
        
    except Exception as e:
        error_msg = f"处理图像路径时发生错误: {input_file} - {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg


# 配置静态文件服务 - 在所有API路由定义之后挂载，避免路由冲突

app.mount(
    "/results/photographs",
    StaticFiles(directory=str(PHOTOGRAPHS_DIR)),
    name="photographs",
)
app.mount(
    "/results/predicts", StaticFiles(directory=str(PREDICTS_DIR)), name="predicts"
)
app.mount("/results/objects", StaticFiles(directory=str(OBJECTS_DIR)), name="objects")
app.mount(
    "/results/objects_image",
    StaticFiles(directory=str(OBJECTS_IMAGE_DIR)),
    name="objects_image",
)
app.mount(
    "/results/objects_search",
    StaticFiles(directory=str(OBJECTS_SEARCH_DIR)),
    name="objects_search",
)
app.mount(
    "/results/history_image",
    StaticFiles(directory=str(HISTORY_IMAGE_DIR)),
    name="history_image",
)
app.mount(
    "/results/history_search",
    StaticFiles(directory=str(HISTORY_SEARCH_DIR)),
    name="history_search",
)
app.mount("/results/uav_way", StaticFiles(directory=str(UAV_WAY_DIR)), name="uav_way")

# 挂载遥感影像数据目录
app.mount(
    "/data/RS_images", StaticFiles(directory=str(RS_IMAGES_DIR)), name="rs_images"
)

# 挂载MCP服务器
# 创建MCP实例
mcp = FastApiMCP(
    app,
    name="UAV_tools_mcp_server",
    description="UAV image processing API MCP",
    include_operations=[
        # 无人机侦察接口
        "uav_trigger",  # 无人机区域绘制
        "uav_planner",  # 无人机跟踪
        "img_predictor",  # 图像预测
        "img_cropper",  # 图像裁剪
        "objects_searcher",  # 目标图像搜索
        "history_searcher",  # 历史图像搜索
        "vehiRoute_analysis",  # 车辆路径分析

        # 通用、广播接口
        "img_matcher",  # 图像匹配
        "broadcast_default",  # 默认广播
        "broadcast_equipData",  # 装备数据广播
        "broadcast_uavPoint",  # 无人机航点位置信息数据广播
        "broadcast_RSimage",  # 遥感影像数据广播
    ],
)

# Mount the MCP server directly to your app
mcp.mount()

if __name__ == "__main__":

    # 启动FastAPI-MCP服务器 with Socket.IO
    print("🚀 启动API服务/Socket.IO/MCP")
    print(f"📍API文档地址: http://localhost:{PORT}/docs")
    print(f"API根地址: http://localhost:{PORT}")

    # 使用Socket.IO ASGI应用启动服务器
    uvicorn.run(socket_app, host="0.0.0.0", port=PORT)
