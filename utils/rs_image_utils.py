"""
卫星数据处理工具模块
包含卫星列表获取、API请求、MySQL数据库存储和影像下载功能
"""

import json
import requests
import time
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

from pprint import pprint
from pathlib import Path
import pymysql
from pymysql.cursors import DictCursor
import traceback

from shapely.geometry import shape, box
from PIL import Image
import hashlib


def is_valid_image(image_path: str, min_file_size: int = 20000, min_unique_colors: int = 50) -> bool:
    """
    验证下载的图像是否为有效图像（非空白图像）
    
    Args:
        image_path: 图像文件路径
        min_file_size: 最小文件大小（字节），默认20000字节（约20KB）
        min_unique_colors: 最小唯一颜色数，默认50（空白图像通常只有1-2种颜色）
    
    Returns:
        bool: 如果是有效图像返回True，否则返回False
    """
    try:
        # 检查文件是否存在
        if not Path(image_path).exists():
            print(f"⚠️ 文件不存在: {image_path}")
            return False
        
        # 检查文件大小（空白图像通常很小）
        file_size = Path(image_path).stat().st_size
        if file_size < min_file_size:
            print(f"⚠️ 文件过小 ({file_size} bytes < {min_file_size}): {image_path}")
            return False
        
        # # 尝试打开图像验证完整性
        # try:
        #     with Image.open(image_path) as img:
        #         img.verify()  # 验证图像完整性
                
        #     # 重新打开检查内容（verify后需要重新打开）
        #     with Image.open(image_path) as img:
        #         # 检查图像尺寸
        #         width, height = img.size
        #         if width < 100 or height < 100:
        #             print(f"⚠️ 图像尺寸过小 ({width}x{height}): {image_path}")
        #             return False
                
        #         # 检查是否为纯色/空白图像
        #         # 方法1: 采样多个区域检查颜色多样性
        #         img_array = list(img.getdata())
        #         total_pixels = len(img_array)
                
        #         # 采样至少5000个像素或全部像素（取较小值）
        #         sample_size = min(5000, total_pixels)
        #         sampled_pixels = img_array[:sample_size]
        #         unique_colors = len(set(sampled_pixels))
                
        #         # 计算唯一颜色占比
        #         color_ratio = unique_colors / sample_size if sample_size > 0 else 0
                
        #         # 判断标准：唯一颜色数量要足够多，或者颜色占比要合理
        #         if unique_colors < min_unique_colors:
        #             print(f"⚠️ 图像颜色单一 (unique_colors={unique_colors} < {min_unique_colors}): {image_path}")
        #             return False
                
        #         # 额外检查：如果是单一颜色（纯色图像），直接拒绝
        #         if unique_colors <= 3:
        #             print(f"⚠️ 检测到纯色图像 (unique_colors={unique_colors}): {image_path}")
        #             return False
                
        #         # 检查颜色分布是否过于单一（超过95%像素是同一颜色）
        #         if sample_size > 100:
        #             from collections import Counter
        #             color_counts = Counter(sampled_pixels)
        #             most_common_color, most_common_count = color_counts.most_common(1)[0]
        #             dominant_ratio = most_common_count / sample_size
                    
        #             if dominant_ratio > 0.95:
        #                 print(f"⚠️ 图像单一颜色占比过高 (dominant_ratio={dominant_ratio:.2%}): {image_path}")
        #                 return False
        
        # except Exception as e:
        #     print(f"⚠️ 图像验证失败: {image_path}, 错误: {str(e)}")
        #     return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ 验证图像时出错: {str(e)}")
        return False


def is_polygon_intersects_bbox(geojson: Dict[str, Any], min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
    """
    检查 GeoJSON Polygon 是否与给定的矩形边界框在空间上有重叠（面积交集）
    
    使用 Shapely 库进行真正的几何重叠判断，而不是简单的边界框相交判断。
    
    Args:
        geojson: GeoJSON 对象，格式如 {"type":"Polygon","coordinates":[[[lon,lat],...]]}
        min_lon: 边界框最小经度（西）
        min_lat: 边界框最小纬度（南）
        max_lon: 边界框最大经度（东）
        max_lat: 边界框最大纬度（北）
    
    Returns:
        bool: 如果在空间上有重叠返回 True，否则返回 False
    """
    try:
        # 将 GeoJSON 转换为 Shapely Polygon 对象
        polygon = shape(geojson)
        
        # 创建 extent 矩形
        extent_box = box(min_lon, min_lat, max_lon, max_lat)
        
        # 判断是否有空间重叠（面积交集）
        # intersects() 方法会检查两个几何体是否有任何空间上的交集
        return polygon.intersects(extent_box)
        
    except Exception as e:
        print(f"⚠️ 检查 Polygon 空间重叠时出错: {str(e)}")
        return False


def calculate_intersection_ratio(geojson: Dict[str, Any], min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> float:
    """
    计算 GeoJSON Polygon 与给定矩形边界框的相交面积占边界框面积的比例
    
    Args:
        geojson: GeoJSON 对象，格式如 {"type":"Polygon","coordinates":[[[lon,lat],...]]}
        min_lon: 边界框最小经度（西）
        min_lat: 边界框最小纬度（南）
        max_lon: 边界框最大经度（东）
        max_lat: 边界框最大纬度（北）
    
    Returns:
        float: 相交面积占边界框面积的比例 (0-1)，如果没有相交则返回0
            - 0: 完全不相交
            - 0.5: 相交面积占extent的50%
            - 1.0: 完全包含或相等
    
    Examples:
        >>> boundary = {"type":"Polygon","coordinates":[[[-118.767,34.483],[-117.349,34.245],[-117.608,33.262],[-119.01,33.499],[-118.767,34.483]]]}
        >>> ratio = calculate_intersection_ratio(boundary, -118.5, 33.5, -117.5, 34.5)
        >>> print(f"相交占比: {ratio:.2%}")  # 输出如: 相交占比: 75.32%
    """
    try:
        # 将 GeoJSON 转换为 Shapely Polygon 对象
        polygon = shape(geojson)
        
        # 创建 extent 矩形
        extent_box = box(min_lon, min_lat, max_lon, max_lat)
        
        # 计算 extent 的面积
        extent_area = extent_box.area
        
        if extent_area == 0:
            return 0.0
        
        # 计算相交区域
        intersection = polygon.intersection(extent_box)
        
        # 计算相交面积
        intersection_area = intersection.area
        
        # 计算占比（0-1之间的小数）
        ratio = intersection_area / extent_area
        
        return ratio
        
    except Exception as e:
        print(f"⚠️ 计算相交面积占比时出错: {str(e)}")
        return 0.0


def get_satellite_name_list() -> List[Dict[str, Any]]:
    """
    卫星列表获取工具 - 返回扩展的卫星列表，包含卫星ID和对应的传感器ID列表

    触发关键词: 卫星列表、传感器、获取列表、卫星信息

    Returns:
        List[Dict[str, Any]]: 包含satelliteId和sensorIds的卫星列表
    """
    extended_satellites = [
        {"satelliteId": "ZY3-1", "sensorIds": ["MUX", "NAD", "FWD", "BWD"]},
        {"satelliteId": "ZY3-2", "sensorIds": ["MUX", "NAD", "FWD", "BWD"]},
        {"satelliteId": "ZY3-3", "sensorIds": ["MUX", "NAD", "FWD", "BWD"]},
        {"satelliteId": "ZY02C", "sensorIds": ["HRC", "PMS"]},
        {"satelliteId": "ZY1E", "sensorIds": ["VNIC"]},
        {"satelliteId": "ZY1F", "sensorIds": ["VNIC"]},
        {"satelliteId": "GF1", "sensorIds": ["PMS1", "PMS2"]},
        {"satelliteId": "GF6", "sensorIds": ["PMS"]},
        {"satelliteId": "CB04A", "sensorIds": ["WPM"]},
        {"satelliteId": "GF2", "sensorIds": ["PMS1", "PMS2"]},
        {"satelliteId": "GF7-01", "sensorIds": ["MUX", "BWD"]},
        {"satelliteId": "GFDM01", "sensorIds": ["PMS"]},
        {
            "satelliteId": "JL1",
            "sensorIds": [
                "PMS",
                "PMS1",
                "PMS2",
                "PMS01",
                "PMS02",
                "PMS03",
                "PMS04",
                "PMS05",
                "PMS06",
                "PMSL1",
                "PMSL2",
                "PMSL3",
                "PMSL4",
                "PMSL5",
                "PMSL6",
                "PMSR1",
                "PMSR2",
                "PMSR3",
                "PMSR4",
                "PMSR5",
                "PMSR6",
            ],
        },
        {"satelliteId": "BJ3", "sensorIds": ["PMS", "PMS1", "PMS2"]},
        {"satelliteId": "BJ2", "sensorIds": ["MS"]},
        {"satelliteId": "SV1", "sensorIds": ["PMS"]},
        {"satelliteId": "GE01", "sensorIds": ["PMS"]},
        {"satelliteId": "WV02", "sensorIds": ["PMS"]},
        {"satelliteId": "Pleiades", "sensorIds": ["PMS"]},
        {"satelliteId": "K2", "sensorIds": ["PMS"]},
        {"satelliteId": "K3", "sensorIds": ["PMS"]},
        {"satelliteId": "DEIMOS", "sensorIds": ["PMS"]},
        {"satelliteId": "WV03", "sensorIds": ["PMS"]},
        {"satelliteId": "WV04", "sensorIds": ["PMS"]},
        {"satelliteId": "K3A", "sensorIds": ["PMS"]},
        {"satelliteId": "SV2", "sensorIds": ["PMS"]},
        {"satelliteId": "GF4", "sensorIds": ["PMI", "IRS"]},
        {"satelliteId": "Landsat-8", "sensorIds": ["OLI_TIRS"]},
        {"satelliteId": "GF5", "sensorIds": ["AHSI", "VIMS"]},
        {"satelliteId": "GF5A", "sensorIds": ["AHSI"]},
        {"satelliteId": "GF5B", "sensorIds": ["AHSI"]},
        {"satelliteId": "GF3", "sensorIds": ["SL"]},
        {"satelliteId": "GF3B", "sensorIds": ["SL"]},
        {"satelliteId": "TerraSAR-X", "sensorIds": ["ST"]},
        {"satelliteId": "GF1B", "sensorIds": ["PMS"]},
        {"satelliteId": "GF1C", "sensorIds": ["PMS"]},
        {"satelliteId": "GF1D", "sensorIds": ["PMS"]},
        {"satelliteId": "TH01-01", "sensorIds": ["GFB", "DGP"]},
        {"satelliteId": "TH01-02", "sensorIds": ["GFB", "DGP"]},
        {"satelliteId": "TH01-03", "sensorIds": ["GFB", "DGP"]},
        {"satelliteId": "TH01-04", "sensorIds": ["GFB", "DGP"]},
        {"satelliteId": "JL1KF01B", "sensorIds": ["PMS"]},
        {"satelliteId": "JL1KF02B", "sensorIds": ["PMS"]},
        {"satelliteId": "JL1KF01C", "sensorIds": ["PMS"]},
        {"satelliteId": "JL1GF04A", "sensorIds": ["PMS"]},
        {"satelliteId": "Landsat-9", "sensorIds": ["OLI_TIRS"]},
        {"satelliteId": "JL1GP01", "sensorIds": ["PMS"]},
        {"satelliteId": "JL1GP02", "sensorIds": ["PMS"]},
        {"satelliteId": "OHS-2A", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-2B", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-2C", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-2D", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-3A", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-3B", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-3C", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OHS-3D", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "LT1A", "sensorIds": ["STRIP1"]},
        {"satelliteId": "LT1B", "sensorIds": ["STRIP1"]},
        {"satelliteId": "GF3C", "sensorIds": ["SL"]},
        {"satelliteId": "OVS-2A", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
        {"satelliteId": "OVS-3A", "sensorIds": ["CMOS", "CMOSMSS", "CMOS-HS"]},
    ]

    return extended_satellites


# def get_satellite_metadata(
#     time_start: str,
#     time_end: str,
#     extent: List[float],
#     satellite_list: Optional[List[Dict[str, Any]]] = None,
#     cloud_percent_min: int = 0,
#     cloud_percent_max: int = 30,
# ) -> List[str]:
#     """
#     卫星元数据查询工具 - 向卫星元数据API发送POST请求获取卫星影像数据

#     触发关键词: 卫星、影像、元数据、查询、搜索、获取

#     Args:
#         time_start (str): 获取时间开始，支持字符串格式：
#             - "2024-01-01 12:00:00" (完整时间格式)
#             - "2024-01-01" (日期格式)
#         time_end (str): 获取时间结束，支持格式（同上）
#         extent (List[float]): 地理范围 [经度1, 纬度1, 经度2, 纬度2]
#         satellite_list (Optional[List[Dict[str, Any]]]): 卫星列表，如果为None则使用默认列表
#         cloud_percent_min (int): 最小云量百分比，默认0
#         cloud_percent_max (int): 最大云量百分比，默认20

#     Returns:
#         List[str]: 卫星影像的quickViewUri列表

#     Examples:
#         示例调用（支持字符串时间格式）:

#         # 使用完整时间字符串格式
#         send_satellite_metadata_request(
#             time_start="2025-09-01 00:00:00",
#             time_end="2025-09-30 23:59:59",
#             extent=[120.866, 37.602, 120.866, 37.602]
#         )

#         # 使用日期字符串格式
#         send_satellite_metadata_request(
#             time_start="2025-09-01",
#             time_end="2025-09-30",
#             extent=[120.866, 37.602, 120.866, 37.602]
#         )
#     """

#     # API接口地址
#     api_url = "http://114.116.226.59/api/normal/v5/normalmeta"

#     # 参数验证
#     if not isinstance(extent, list) or len(extent) != 4:
#         raise ValueError(
#             "extent参数必须是包含4个元素的列表 [左上经度（西）, 左上纬度（北）, 右下经度（东）, 右下纬度（南）]"
#         )

#     # 时间格式转换 - 将字符串格式转换为毫秒时间戳
#     try:
#         start_timestamp = calculate_millisecond_timestamp(time_start)
#         end_timestamp = calculate_millisecond_timestamp(time_end)
#     except Exception as e:
#         raise ValueError(f"时间格式转换失败: {e}")

#     # 验证时间范围
#     if start_timestamp >= end_timestamp:
#         raise ValueError("time_start必须小于time_end")

#     # 如果没有提供卫星列表，使用默认列表
#     if satellite_list is None:
#         satellite_list = get_satellite_name_list()

#     # 构建请求参数
#     request_data = {
#         "acquisitionTime": [{"Start": start_timestamp, "End": end_timestamp}],
#         "tarInputTimeStart": None,
#         "tarInputTimeEnd": None,
#         "inputTimeStart": None,
#         "inputTimeEnd": None,
#         "cloudPercentMin": cloud_percent_min,
#         "cloudPercentMax": cloud_percent_max,
#         "satellite_list": satellite_list,
#         "extent": extent,
#         "pageNum": 1,
#         "pageSize": 20,
#     }

#     # 设置请求头
#     headers = {"Content-Type": "application/json", "Accept": "application/json"}

#     try:
#         # 发送POST请求
#         response = requests.post(
#             api_url, json=request_data, headers=headers, timeout=30
#         )
#         # 检查响应状态码
#         response.raise_for_status()

#         # response1 = requests.post(
#         #     "http://localhost:8080/api/push-data",
#         #     json=response.json(),
#         #     headers=headers,
#         #     timeout=30
#         # )
#         # # 检查响应状态码
#         # response1.raise_for_status()
#         # print("*******************success***************")
#         # print(response1.json())

#         data = response.json()["data"]
#         print(f"获取到 {len(data)} 条卫星数据")

#         # 返回JSON响应
#         return data

#         # data = response.json()["data"]
#         # image_list = [i["quickViewUri"] for i in data]
#         # return image_list

#     except requests.exceptions.Timeout:
#         raise requests.RequestException("请求超时（30秒）")
#     except requests.exceptions.ConnectionError:
#         raise requests.RequestException("连接错误，请检查网络连接")
#     except requests.exceptions.HTTPError as e:
#         raise requests.RequestException(f"HTTP错误: {e}")
#     except requests.exceptions.RequestException as e:
#         raise requests.RequestException(f"请求失败: {e}")
#     except json.JSONDecodeError:
#         raise requests.RequestException("响应不是有效的JSON格式")



def get_satellite_metadata(
    time_start: str,
    time_end: str,
    extent: List[float],
    satellite_list: Optional[List[Dict[str, Any]]] = None,
    cloud_percent_min: int = 0,
    cloud_percent_max: int = 15,
    max_results: Optional[int] = 30,  # 新增：最大返回结果数，None表示获取所有
    min_intersection_ratio: float = 0.0,  # 新增：最小相交面积占比（0-1），默认0表示不过滤
) -> List[str]:
    """
    卫星元数据查询工具 - 向卫星元数据API发送POST请求获取卫星影像数据，支持相交面积占比过滤
    
    触发关键词: 卫星、影像、元数据、查询、搜索、获取
    
    Args:
        time_start (str): 获取时间开始，支持字符串格式：
            - "2024-01-01 12:00:00" (完整时间格式)
            - "2024-01-01" (日期格式)
        time_end (str): 获取时间结束，支持格式（同上）
        extent (List[float]): 地理范围 [左上经度（西）, 左上纬度（北）, 右下经度（东）, 右下纬度（南）]
        satellite_list (Optional[List[Dict[str, Any]]]): 卫星列表，如果为None则使用默认列表
        cloud_percent_min (int): 最小云量百分比，默认0
        cloud_percent_max (int): 最大云量百分比，默认15
        max_results (Optional[int]): 最大返回结果数，None表示获取所有
        min_intersection_ratio (float): 最小相交面积占比（0-1之间的小数），
            表示boundary与extent相交区域面积占extent面积的最小比例，默认0表示不过滤
            - 0: 不过滤（默认）
            - 0.5: 相交面积至少占extent的50%
            - 0.8: 相交面积至少占extent的80%
            - 1.0: 完全覆盖
    
    Returns:
        List[Dict]: 卫星影像数据列表
    
    Examples:
        # 基本使用
        data = get_satellite_metadata(
            time_start="2025-09-01",
            time_end="2025-09-30",
            extent=[120.866, 37.602, 121.866, 38.602]
        )
        
        # 使用相交占比过滤（只保留相交面积>=50%的数据）
        data = get_satellite_metadata(
            time_start="2025-09-01",
            time_end="2025-09-30",
            extent=[120.866, 37.602, 121.866, 38.602],
            min_intersection_ratio=0.5  # 0.5 表示 50%
        )
        
        # 严格过滤（只保留相交面积>=80%的数据）
        data = get_satellite_metadata(
            time_start="2025-09-01",
            time_end="2025-09-30",
            extent=[120.866, 37.602, 121.866, 38.602],
            min_intersection_ratio=0.8  # 0.8 表示 80%
        )
    """
    
    api_url = "http://114.116.226.59/api/normal/v5/normalmeta"
    
    # 参数验证
    if not isinstance(extent, list) or len(extent) != 4:
        raise ValueError(
            "extent参数必须是包含4个元素的列表 [左上经度（西）, 左上纬度（北）, 右下经度（东）, 右下纬度（南）]"
        )

    # 时间格式转换 - 将字符串格式转换为毫秒时间戳
    try:
        start_timestamp = calculate_millisecond_timestamp(time_start)
        end_timestamp = calculate_millisecond_timestamp(time_end)
    except Exception as e:
        raise ValueError(f"时间格式转换失败: {e}")

    # 验证时间范围
    if start_timestamp >= end_timestamp:
        raise ValueError("time_start必须小于time_end")

    # 如果没有提供卫星列表，使用默认列表
    if satellite_list is None:
        satellite_list = get_satellite_name_list()
    
    all_data = []
    filtered_data = []  # 存储相交占比过滤后的数据
    page_num = 1
    page_size = 50  # 可以设置较大的每页数量
    
    # 判断是否需要相交占比过滤
    need_ratio_filter = min_intersection_ratio > 0
    if need_ratio_filter:
        min_lon, min_lat, max_lon, max_lat = extent[0], extent[1], extent[2], extent[3]
        print(f"\n🔍 已启用相交面积占比过滤（最小占比: {min_intersection_ratio:.1%}）")
    
    while True:
        # 构建请求参数
        request_data = {
            "acquisitionTime": [{"Start": start_timestamp, "End": end_timestamp}],
            "tarInputTimeStart": None,
            "tarInputTimeEnd": None,
            "inputTimeStart": None,
            "inputTimeEnd": None,
            "cloudPercentMin": cloud_percent_min,
            "cloudPercentMax": cloud_percent_max,
            "satellite_list": satellite_list,
            "extent": extent,
            "pageNum": page_num,
            "pageSize": page_size,
        }
        
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        try:
            response = requests.post(
                api_url, json=request_data, headers=headers, timeout=30
            )
            response.raise_for_status()
            
            data = response.json()["data"]
            
            # 如果没有数据或数据为空，说明已经获取完所有页
            if not data:
                break
            
            all_data.extend(data)
            
            print(f"已获取第 {page_num} 页，本页 {len(data)} 条，累计 {len(all_data)} 条")
            
            # 步骤1: 先进行相交占比过滤（如果启用）
            if need_ratio_filter:
                for item in data:
                    boundary_str = item.get("boundary")
                    if boundary_str:
                        try:
                            # 解析 boundary GeoJSON
                            boundary_geojson = json.loads(boundary_str) if isinstance(boundary_str, str) else boundary_str
                            
                            # 计算相交面积占比
                            intersection_ratio = calculate_intersection_ratio(
                                boundary_geojson, min_lon, min_lat, max_lon, max_lat
                            )
                            
                            # 如果占比满足条件，保留该数据
                            if intersection_ratio >= min_intersection_ratio:
                                filtered_data.append(item)
                                print(f"  ID={item.get('id')} 相交占比={intersection_ratio:.1%} ✓ 保留（累计过滤后: {len(filtered_data)} 条）")
                            else:
                                print(f"  ID={item.get('id')} 相交占比={intersection_ratio:.1%} ✗ 过滤")
                                
                        except Exception as e:
                            print(f"  ⚠️ 解析 boundary 失败 (ID: {item.get('id')}): {str(e)}")
                            # 解析失败的记录默认不保留
                            continue
                    else:
                        print(f"  ⚠️ ID={item.get('id')} 没有 boundary 数据，跳过")
            
            # 如果本页数据少于pageSize，说明这是最后一页
            if len(data) < page_size:
                break
            
            # 步骤2: 再应用最大结果数限制（在过滤后的数据上）
            if max_results:
                current_count = len(filtered_data) if need_ratio_filter else len(all_data)
                if current_count >= max_results:
                    print(f"\n已达到最大结果数限制 ({max_results} 条)，停止获取")
                    break
            
            page_num += 1
            
        except requests.exceptions.Timeout:
            raise requests.RequestException("请求超时（30秒）")
        except requests.exceptions.ConnectionError:
            raise requests.RequestException("连接错误，请检查网络连接")
        except requests.exceptions.HTTPError as e:
            raise requests.RequestException(f"HTTP错误: {e}")
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"请求失败: {e}")
        except json.JSONDecodeError:
            raise requests.RequestException("响应不是有效的JSON格式")
    
    print(f"\n总共获取到 {len(all_data)} 条原始卫星数据")
    
    # 返回结果
    if need_ratio_filter:
        # 应用 max_results 限制（在已过滤的数据上）
        if max_results and len(filtered_data) > max_results:
            filtered_data = filtered_data[:max_results]
            print(f"应用最大结果数限制: {len(filtered_data)} 条")
        
        print(f"✅ 最终结果: 原始 {len(all_data)} 条 → 过滤后 {len(filtered_data)} 条")
        return filtered_data
    else:
        # 如果没有相交占比过滤，直接应用 max_results 限制
        if max_results and len(all_data) > max_results:
            all_data = all_data[:max_results]
            print(f"应用最大结果数限制: {len(all_data)} 条")
        return all_data


def calculate_millisecond_timestamp(
    input_time: Union[str, datetime, None] = None,
) -> int:
    """
    计算毫秒时间戳

    Args:
        input_time: 输入时间，支持以下格式：
            - None: 返回当前时间的毫秒时间戳
            - str: 支持多种字符串格式，如 "2024-01-01 12:00:00", "2024-01-01", "2024/01/01 12:00:00"
            - datetime: datetime对象

    Returns:
        int: 毫秒时间戳（13位数字）

    Examples:
        >>> calculate_millisecond_timestamp()  # 当前时间
        1704067200000
        >>> calculate_millisecond_timestamp("2024-01-01 12:00:00")
        1704067200000
        >>> calculate_millisecond_timestamp("2024-01-01")
        1704067200000
    """
    if input_time is None:
        # 返回当前时间的毫秒时间戳
        return int(time.time() * 1000)

    if isinstance(input_time, str):
        # 处理字符串输入
        try:
            # 尝试多种日期格式
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d",
                "%Y.%m.%d %H:%M:%S",
                "%Y.%m.%d %H:%M",
                "%Y.%m.%d",
                "%Y-%m-%d %H:%M:%S.%f",  # 支持微秒
            ]

            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(input_time, fmt)
                    break
                except ValueError:
                    continue

            if dt is None:
                raise ValueError(f"无法解析时间格式: {input_time}")

            # 如果没有时区信息，假设为本地时区
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return int(dt.timestamp() * 1000)

        except Exception as e:
            raise ValueError(f"时间字符串解析失败: {input_time}, 错误: {str(e)}")

    elif isinstance(input_time, datetime):
        # 处理datetime对象
        if input_time.tzinfo is None:
            input_time = input_time.replace(tzinfo=timezone.utc)
        return int(input_time.timestamp() * 1000)

    else:
        raise TypeError(f"不支持的时间类型: {type(input_time)}")


def timestamp_to_datetime(timestamp: int, use_local_time: bool = True) -> datetime:
    """
    将毫秒时间戳转换为datetime对象

    Args:
        timestamp: 毫秒时间戳
        use_local_time: 是否使用本地时间，False则使用UTC时间

    Returns:
        datetime: 对应的datetime对象
    """
    # 将毫秒时间戳转换为秒
    seconds = timestamp / 1000

    if use_local_time:
        return datetime.fromtimestamp(seconds)
    else:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)


def save_metadata_to_mysql(
    # 卫星数据参数
    satellite_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    downloaded_files_map: Dict[int, str] = None,  # 新增：已下载文件的映射 {id: local_file_path}
    # MySQL存储参数
    host: str = "localhost",
    port: int = 3306,
    user: str = "root", 
    password: str = "123456",
    database: str = "RS_images_db",
    table_name: str = "RS_images_metadata",
    auto_create_db: bool = False,
    auto_create_table: bool = True,
) -> Dict[str, Any]:
    """
    将卫星数据写入MySQL数据库

    触发关键词: 保存、存储、写入、数据库、MySQL

    Args:
        satellite_data: 卫星数据，可以是单个字典或字典列表（从send_satellite_metadata_request返回的data字段）
        downloaded_files_map: 已下载文件的映射字典 {id: local_file_path}
        host: MySQL服务器地址，默认localhost
        port: MySQL端口，默认3306
        user: MySQL用户名，默认root
        password: MySQL密码，默认123456
        database: 数据库名称，默认RS_images_db
        table_name: 表名称，默认RS_images_metadata
        auto_create_db: 是否自动创建数据库，默认False
        auto_create_table: 是否自动创建表，默认True

    Returns:
        Dict[str, Any]: 包含执行结果的字典
            - success: bool, 是否成功
            - inserted_count: int, 插入的记录数
            - updated_count: int, 更新的记录数
            - message: str, 执行消息
            - errors: List[str], 错误信息列表

    Examples:
        # 先下载图像，然后保存元数据
        files_map = {id1: "/path/to/image1.jpg", id2: "/path/to/image2.jpg"}
        result = save_satellite_metadata_to_mysql(
            satellite_data=data_list,
            downloaded_files_map=files_map
        )
    """
    
    connection = None
    result = {
        "success": False,
        "inserted_count": 0,
        "updated_count": 0,
        "message": "",
        "errors": [],
    }

    try:
        # 统一处理输入数据格式
        if isinstance(satellite_data, dict):
            # 如果是API响应格式，提取data字段
            if "data" in satellite_data:
                data_list = satellite_data["data"]
            else:
                data_list = [satellite_data]
        elif isinstance(satellite_data, list):
            data_list = satellite_data
        else:
            raise ValueError(f"不支持的数据类型: {type(satellite_data)}")

        if not data_list:
            result["message"] = "没有数据需要写入"
            result["success"] = True
            return result

        # 如果需要自动创建数据库，先连接到MySQL服务器
        if auto_create_db:
            temp_connection = pymysql.connect(
                host=host, port=port, user=user, password=password, charset="utf8mb4"
            )
            try:
                with temp_connection.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                temp_connection.commit()
                print(f"✓ 数据库 '{database}' 已确认存在")
            finally:
                temp_connection.close()

        # 连接到指定数据库
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )

        # 如果需要，创建表
        if auto_create_table:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` INT NOT NULL,
                `acquisitionTime` BIGINT NULL COMMENT '获取时间(毫秒时间戳)',
                `baseId` VARCHAR(100) NULL COMMENT '基础ID',
                `boundary` TEXT NULL COMMENT '边界信息(GeoJSON)',
                `cloudPercent` INT NULL COMMENT '云量百分比',
                `filename` VARCHAR(255) NULL COMMENT '文件名',
                `hasEntity` TINYINT NULL COMMENT '是否有实体',
                `hasPair` TINYINT NULL COMMENT '是否有配对',
                `inCart` VARCHAR(50) NULL COMMENT '是否在购物车',
                `inputTime` BIGINT NULL COMMENT '输入时间(毫秒时间戳)',
                `laserCount` VARCHAR(50) NULL COMMENT '激光计数',
                `orbitId` INT NULL COMMENT '轨道ID',
                `productId` VARCHAR(100) NULL COMMENT '产品ID',
                `quickViewUri` TEXT NULL COMMENT '快速预览URI',
                `localFile` TEXT NULL COMMENT '本地文件路径',
                `satelliteId` VARCHAR(50) NULL COMMENT '卫星ID',
                `scenePath` INT NULL COMMENT '场景路径',
                `sceneRow` INT NULL COMMENT '场景行',
                `sensorId` VARCHAR(50) NULL COMMENT '传感器ID',
                `tarInputTime` BIGINT NULL COMMENT 'Tar输入时间(毫秒时间戳)',
                PRIMARY KEY (`id`),
                INDEX `idx_satelliteId` (`satelliteId`),
                INDEX `idx_acquisitionTime` (`acquisitionTime`),
                INDEX `idx_orbitId` (`orbitId`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='卫星影像元数据表';
            """

            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                connection.commit()
                print(f"✓ 数据表 '{table_name}' 已确认存在")


        # 使用外部提供的下载文件映射（如果没有提供则设为空字典）
        if downloaded_files_map is None:
            downloaded_files_map = {}

        # 准备插入/更新数据
        with connection.cursor() as cursor:
            for idx, item in enumerate(data_list):
                try:
                    # 从映射中获取本地文件路径
                    item_id = item.get("id")
                    saved_file = downloaded_files_map.get(item_id)
                    
                    # 提取文件名
                    if saved_file:
                        img_file_name = Path(saved_file).name
                    else:
                        img_file_name = None
                    
                    # 使用INSERT ... ON DUPLICATE KEY UPDATE语法实现upsert
                    insert_sql = f"""
                    INSERT INTO `{table_name}` (
                        id, acquisitionTime, baseId, boundary, cloudPercent,
                        filename, hasEntity, hasPair, inCart, inputTime,
                        laserCount, orbitId, productId, quickViewUri, localFile,
                        satelliteId, scenePath, sceneRow, sensorId, tarInputTime
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        acquisitionTime = VALUES(acquisitionTime),
                        baseId = VALUES(baseId),
                        boundary = VALUES(boundary),
                        cloudPercent = VALUES(cloudPercent),
                        filename = VALUES(filename),
                        hasEntity = VALUES(hasEntity),
                        hasPair = VALUES(hasPair),
                        inCart = VALUES(inCart),
                        inputTime = VALUES(inputTime),
                        laserCount = VALUES(laserCount),
                        orbitId = VALUES(orbitId),
                        productId = VALUES(productId),
                        quickViewUri = VALUES(quickViewUri),
                        localFile = VALUES(localFile),
                        satelliteId = VALUES(satelliteId),
                        scenePath = VALUES(scenePath),
                        sceneRow = VALUES(sceneRow),
                        sensorId = VALUES(sensorId),
                        tarInputTime = VALUES(tarInputTime)
                    """

                    values = (
                        item.get("id"),
                        item.get("acquisitionTime"),
                        item.get("baseId"),
                        item.get("boundary"),
                        item.get("cloudPercent"),
                        img_file_name,
                        item.get("hasEntity"),
                        item.get("hasPair"),
                        item.get("inCart"),
                        item.get("inputTime"),
                        item.get("laserCount"),
                        item.get("orbitId"),
                        item.get("productId"),
                        item.get("quickViewUri"),
                        saved_file,                 # localFile 字段
                        item.get("satelliteId"),
                        item.get("scenePath"),
                        item.get("sceneRow"),
                        item.get("sensorId"),
                        item.get("tarInputTime"),
                    )

                    affected_rows = cursor.execute(insert_sql, values)

                    # 判断是插入还是更新
                    if affected_rows == 1:
                        result["inserted_count"] += 1
                    elif affected_rows == 2:
                        result["updated_count"] += 1

                except Exception as e:
                    error_msg = f"处理第{idx+1}条数据时出错 (ID: {item.get('id', 'unknown')}): {str(e)}"
                    result["errors"].append(error_msg)
                    print(f"✗ {error_msg}")

            # 提交事务
            connection.commit()

        # 设置成功状态
        total_processed = result["inserted_count"] + result["updated_count"]
        result["success"] = True
        result["message"] = (
            f"成功处理 {total_processed} 条数据 (插入: {result['inserted_count']}, 更新: {result['updated_count']})"
        )

        if result["errors"]:
            result["message"] += f", 失败: {len(result['errors'])} 条"

        print(f"✓ {result['message']}")

    except pymysql.Error as e:
        error_msg = f"MySQL错误: {str(e)}"
        result["errors"].append(error_msg)
        result["message"] = error_msg
        print(f"✗ {error_msg}")

    except Exception as e:
        error_msg = f"执行错误: {str(e)}"
        result["errors"].append(error_msg)
        result["message"] = error_msg
        print(f"✗ {error_msg}")
        import traceback

        traceback.print_exc()

    finally:
        # 关闭数据库连接
        if connection:
            connection.close()
            print("✓ 数据库连接已关闭")

    return result


def download_image(
    download_url: str,
    save_dir: str = None,
    skip_existing: bool = True,
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[str]:
    """
    根据URL下载卫星影像到本地

    触发关键词: 下载、影像、图片、保存图片

    Args:
        url: 图像下载URL
        save_dir: 保存目录，默认"./data/RS_images"
        skip_existing: 是否跳过已存在的文件，默认True
        timeout: 下载超时时间（秒），默认60
        max_retries: 最大重试次数，默认3

    Returns:
        Optional[str]: 成功返回保存的文件路径，失败返回None

    Examples:
        url = 'http://quickview.sasclouds.com/LT1B/15632/0/0/LT1B_MONO_SYC_STRIP1_015632.jpg'
        file_path = download_image(url)
        if file_path:
            print(f"文件已保存到: {file_path}")
    """
    # 创建保存目录
    if save_dir is None:
        save_path = Path(__file__).parent.parent / "data/RS_images"
    else:
        save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 从URL提取文件名
    filename = download_url.split("/")[-1]   # 文件名
    if not filename.endswith(".jpg"):
        filename = f"{filename}.jpg"
    
    save_file_path = save_path / filename

    # 检查文件是否已存在
    if skip_existing and save_file_path.exists():
        return str(save_file_path)

    # 下载文件（支持重试）
    for retry in range(max_retries):
        try:
            response = requests.get(download_url, timeout=timeout, stream=True)
            response.raise_for_status()

            # 写入文件
            with open(save_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 验证文件大小
            file_size = save_file_path.stat().st_size
            if file_size == 0:
                raise ValueError("下载的文件大小为0")
            print(f"✓ 完成遥感影像下载，文件已保存到: {save_file_path}")

            return str(save_file_path)

        except Exception:
            if retry < max_retries - 1:
                time.sleep(1)
            else:
                return None

    return None


def get_satellite_from_mysql(
    # 数据查询参数
    acquisitionTime: List[Dict[str, int]] = None, # 影像采集时间 {"Start": start_timestamp,"End": end_timestamp}
    # time_start: str = None,      # 获取时间开始,支持字符串格式
    # time_end: str = None,        # 获取时间结束,支持字符串格式
    extent: List[float] = None,  # 地理范围 [经度1, 纬度1, 经度2, 纬度2]         
    cloud_percent_min: int = 0,  # 最小云量百分比
    cloud_percent_max: int = 20, # 最大云量百分比
    limit: int = None,           # 限制返回记录数,默认None（返回所有）
    # MySQL配置参数
    host: str = "localhost",     # MySQL服务器地址
    port: int = 3306,            # MySQL端口
    user: str = "root",          # MySQL用户名
    password: str = "123456",    # MySQL密码
    database: str = "RS_images_db",         # 数据库名称    
    table_name: str = "RS_images_metadata", # 表名称
) -> Dict[str, Any]:
    """
    从本地MySQL数据库读取卫星元数据

    触发关键词: 读取、查询、本地数据库、获取本地数据

    Args:
        # 数据查询参数
        acquisitionTime (Optional[List[Dict[str, int]]]): 影像采集时间 {"Start": start_timestamp,"End": end_timestamp}
        time_end (Optional[str]): 获取时间结束
        extent (Optional[List[float]]): 地理范围 [经度1, 纬度1, 经度2, 纬度2]
        satellite_ids (Optional[List[str]]): 卫星ID列表，如 ["LT1A", "GF6"]
        cloud_percent_min (Optional[int]): 最小云量百分比
        cloud_percent_max (Optional[int]): 最大云量百分比
        limit (Optional[int]): 限制返回记录数，默认None（返回所有）
        # MySQL配置参数
        host (str): MySQL服务器地址，默认localhost
        port (int): MySQL端口，默认3306
        user (str): MySQL用户名，默认root
        password (str): MySQL密码，默认123456
        database (str): 数据库名称，默认RS_images_db
        table_name (str): 表名称，默认metadata
    Returns:
        Dict[str, Any]: 与send_satellite_metadata_request相同格式的响应
            {
                "data": [
                    {
                        "id": 38993896,
                        "acquisitionTime": 1756591671132,
                        "baseId": None,
                        "boundary": "...",
                        "localFile": "/path/to/images/filename.jpg",  # 如果提供了save_dir
                        ...
                    },
                    ...
                ]
            }

    Examples:
        # 查询所有数据
        result = get_satellite_metadata_from_mysql()

        # 按时间范围查询
        result = get_satellite_metadata_from_mysql(
            time_start="2025-08-01",
            time_end="2025-09-01"
        )

        # 按卫星ID查询
        result = get_satellite_metadata_from_mysql(
            satellite_ids=["LT1A", "GF6"]
        )

        # 组合查询
        result = get_satellite_metadata_from_mysql(
            time_start="2025-08-01",
            time_end="2025-09-01",
            satellite_ids=["LT1A"],
            cloud_percent_max=20,
            limit=10
        )

        # 添加本地文件路径
        result = get_satellite_metadata_from_mysql(
            satellite_ids=["LT1A"],
            save_dir="./satellite_images"
        )
        # 结果中每条记录会包含 localFile 字段
    """

    connection = None

    try:
        # 连接到数据库
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )

        # 构建SQL查询语句
        where_clauses = []
        params = []

        # 调试信息
        print(f"🔍 DEBUG: acquisitionTime = {acquisitionTime}")
        print(f"🔍 DEBUG: acquisitionTime type = {type(acquisitionTime)}")

        # 时间范围过滤
        if acquisitionTime is not None and len(acquisitionTime) > 0:
            # 从列表中提取第一个时间范围
            time_range = acquisitionTime[0]
            print(f"🔍 DEBUG: time_range = {time_range}")
            
            if time_range.get("Start") is not None:
                start_timestamp = time_range["Start"]
                where_clauses.append("acquisitionTime >= %s")
                params.append(start_timestamp)
                print(f"✅ 添加时间起始过滤: acquisitionTime >= {start_timestamp}")

            if time_range.get("End") is not None:
                end_timestamp = time_range["End"]
                where_clauses.append("acquisitionTime <= %s")
                params.append(end_timestamp)
                print(f"✅ 添加时间结束过滤: acquisitionTime <= {end_timestamp}")
        else:
            print(f"⚠️ 未添加时间过滤: acquisitionTime is None or empty")

        # 地理范围过滤（如果提供）
        # 注意：由于 boundary 字段是 TEXT 类型存储的 GeoJSON，我们在查询后进行 Python 过滤
        # 如果需要在数据库层面过滤，需要将 boundary 字段改为 GEOMETRY 类型
        extent_filter = None
        if extent is not None and len(extent) == 4:
            # extent格式: [minX(西), minY(北), maxX(东), maxY(南)]
            extent_filter = extent
            print(f"⚠️ 将在查询后进行地理范围过滤（Python层）: extent={extent}")
            

        # 卫星ID过滤
        # if satellite_ids is not None:
        #     placeholders = ",".join(["%s"] * len(satellite_ids))
        #     where_clauses.append(f"satelliteId IN ({placeholders})")
        #     params.extend(satellite_ids)

        # 云量过滤
        if cloud_percent_min is not None:
            where_clauses.append("cloudPercent >= %s")
            params.append(cloud_percent_min)

        if cloud_percent_max is not None:
            where_clauses.append("cloudPercent <= %s")
            params.append(cloud_percent_max)

        # 构建完整的SQL语句
        sql = f"SELECT * FROM `{table_name}`"

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # 添加排序（按获取时间降序）
        sql += " ORDER BY acquisitionTime DESC"

        # 添加限制
        if limit is not None:
            sql += f" LIMIT {limit}"

        # 打印SQL调试信息
        print(f"\n{'='*80}")
        print(f"📊 SQL查询信息:")
        print(f"  SQL: {sql}")
        print(f"  参数: {params}")
        print(f"{'='*80}\n")

        # 执行查询
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()

        print(f"✓ 从数据库查询到 {len(results)} 条记录")

        # 如果需要地理范围过滤，在 Python 层进行过滤
        if extent_filter is not None:
            filtered_results = []
            min_lon, min_lat, max_lon, max_lat = extent_filter[0], extent_filter[1], extent_filter[2], extent_filter[3]
            
            for item in results:
                boundary_str = item.get("boundary")
                if boundary_str:
                    try:
                        # 解析 GeoJSON
                        boundary_geojson = json.loads(boundary_str) if isinstance(boundary_str, str) else boundary_str
                        
                        # 检查 Polygon 是否与查询范围相交
                        if is_polygon_intersects_bbox(boundary_geojson, min_lon, min_lat, max_lon, max_lat):
                            filtered_results.append(item)
                    except Exception as e:
                        print(f"⚠️ 解析 boundary 失败 (ID: {item.get('id')}): {str(e)}")
                        # 解析失败的记录跳过
                        continue
            
            print(f"✓ 地理范围过滤后剩余 {len(filtered_results)} 条记录")
            return filtered_results
        
        # 返回与API相同的格式
        return results

    except pymysql.Error as e:
        error_msg = f"MySQL错误: {str(e)}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)

    except Exception as e:
        error_msg = f"查询错误: {str(e)}"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        raise Exception(error_msg)

    finally:
        # 关闭数据库连接
        if connection:
            connection.close()


# ==========================================     主函数       ==========================================

def main(
    time_start: str,
    time_end: str,
    extent: List[float],
    satellite_list: Optional[List[Dict[str, Any]]] = None,
    cloud_percent_min: int = 0,
    cloud_percent_max: int = 10,
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "123456",
    database: str = "RS_images_db",
    table_name: str = "RS_images_metadata",
    is_download: bool = True,
    save_dir: str = None,
    max_retries: int = 3,   # 最大重试次数
    max_results: int = 30,  # 最大下载影像数量，None表示获取所有
    min_intersection_ratio: float = 0.7,  # 最小相交面积占比（0-1），默认0.6表示60%
):
    """
    主逻辑函数：下载原始遥感影像并存储有效图像的metadata数据到数据库。
    
    执行流程：
    1. 获取卫星元数据
    2. 根据相交占比过滤数据（可选）
    3. 下载所有影像
    4. 验证影像有效性（过滤空白图像）
    5. 只保存有效影像的元数据到数据库
    
    Args:
        min_intersection_ratio: 最小相交面积占比（0-1之间的小数），
            表示boundary与extent相交区域面积占extent面积的最小比例
            - 0: 不过滤
            - 0.5: 至少50%覆盖
            - 0.6: 至少60%覆盖（默认）
            - 0.8: 至少80%覆盖
    
    触发关键词: 保存、存储、写入、数据库、MySQL、下载、影像
    """
    print("=" * 100)
    print("\n===  原始遥感影像下载 + 有效数据存入MySQL  ===")

    try:
        # 1. 获取卫星数据
        print("\n🛰️  步骤1: 获取卫星元数据...")
        web_result = get_satellite_metadata(
            time_start=time_start,
            time_end=time_end,
            extent=extent,
            satellite_list=satellite_list,
            cloud_percent_min=cloud_percent_min,
            cloud_percent_max=cloud_percent_max,
            max_results=max_results,
            min_intersection_ratio=min_intersection_ratio,
        )

        print(f"✓ 获取到 {len(web_result)} 条卫星数据记录")
        pprint([item['quickViewUri'] for item in web_result])

        # 2. 下载所有影像
        print(f"\n{'-' * 100}\n")
        print("📷 步骤2: 下载所有遥感影像...")
        
        downloaded_files_map = {}  # {id: file_path}
        valid_ids = []  # 存储有效图像的ID列表
        
        for idx, item in enumerate(web_result, 1):
            item_id = item.get('id')
            url = item.get('quickViewUri')
            
            if not url:
                print(f"[{idx}/{len(web_result)}] ⚠️ 记录 ID={item_id} 没有下载链接，跳过")
                continue
            
            print(f"[{idx}/{len(web_result)}] 下载图像 ID={item_id}...")
            downloaded_file = download_image(
                download_url=url,
                save_dir=save_dir,
                skip_existing=True,
                timeout=60,
                max_retries=max_retries,
            )
            
            if downloaded_file:
                downloaded_files_map[item_id] = downloaded_file
                print(f"  → 下载成功: {downloaded_file}")
            else:
                print(f"  → 下载失败")

        # 3. 验证影像有效性
        print(f"\n{'-' * 100}\n")
        print("🔍 步骤3: 验证影像有效性（过滤空白图像）...")
        
        for item_id, file_path in downloaded_files_map.items():
            if is_valid_image(file_path):
                valid_ids.append(item_id)
                print(f"✓ ID={item_id} 图像有效")
            else:
                print(f"✗ ID={item_id} 图像无效（空白或损坏），将被过滤")
        
        print(f"\n验证结果: 下载 {len(downloaded_files_map)} 个图像，有效 {len(valid_ids)} 个")

        # 4. 过滤出有效数据
        print(f"\n{'-' * 100}\n")
        print("📊 步骤4: 过滤有效数据...")
        
        valid_data = [item for item in web_result if item.get('id') in valid_ids]
        valid_files_map = {item_id: downloaded_files_map[item_id] for item_id in valid_ids}
        
        print(f"✓ 过滤后剩余 {len(valid_data)} 条有效记录")

        # 5. 保存到MySQL数据库（只保存有效数据）
        if valid_data:
            print(f"\n{'-' * 100}\n")
            print("💿 步骤5: 保存有效数据到MySQL数据库...")
            
            db_result = save_metadata_to_mysql(
                satellite_data=valid_data,
                downloaded_files_map=valid_files_map,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                table_name=table_name,
            )

            # 显示保存结果
            print(f"\n数据库写入结果:")
            print(f"  成功状态: {db_result['success']}")
            print(f"  插入记录数: {db_result['inserted_count']}")
            print(f"  更新记录数: {db_result['updated_count']}")
            print(f"  执行消息: {db_result['message']}")

            if db_result["errors"]:
                print(f"\n  错误列表:")
                for error in db_result["errors"]:
                    print(f"    - {error}")

            print("\n✅ 完成！有效数据已保存到MySQL数据库")
        else:
            print("\n⚠️ 没有有效数据可以保存到数据库")

    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    
    # ---------  单个地区输入参数下载影像  ----------

    # 上海市："extent":[121.8,30.691701,122.118227,31.0]
    # 洛杉矶："extent": [-118.7109, 34.0061, -117.9987, 34.2530]

    # RS_params = {
    #     "time_start": "2025-01-10 00:00:00",
    #     "time_end": "2025-01-13 23:59:59",
    #     "extent": [-118.7109, 34.0061, -117.9987, 34.2530],
    #     "min_intersection_ratio": 0.3,  # 只保留相交占比>=50%的数据（0.5表示50%）
    #     "max_results": 15,
    # }

    # RS_params = {
    #     "time_start": "2025-08-29 00:00:00",
    #     "time_end": "2025-08-29 23:59:59",
    #     "extent": [121.574012, 22.739218, 123.086970, 24.520513],
    #     "min_intersection_ratio": 0,  # 只保留相交占比>=50%的数据（0.5表示50%）
    #     "max_results": 20
    # }

    # main(**RS_params)  # 传入字典参数


    # ---------  遍历下载区域json文件下载 ---------

    with open('data/热点事件遥感影像下载.json', 'r', encoding='utf-8') as f:
        RS_params = json.load(f)

    for key, value in RS_params.items():
        print(f"下载区域: {key}")
        save_dir = f'data/RS_images_download/{key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        main(**value, save_dir=save_dir,max_results=30,min_intersection_ratio=0)

        print(f"下载区域: {key} 完成")
        

   


