import os
import requests
import pymysql
from pymysql.cursors import DictCursor
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

MIDDLEWARE_URL = os.getenv("MIDDLEWARE_URL", "http://localhost:5000")# 中间件服务API
timeout = 30

# ===================== MCP Server Configuration =====================
mcp = FastMCP(name="本地船舶信息查询工具", port=8202)

# ===================== Database Default Configuration =====================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_NAME = os.getenv("DB_NAME", "shipinfo_db")
DB_TABLE = os.getenv("DB_TABLE", "shipinfo_metadata")


@mcp.tool
def local_shipinfo_search(
    longitude: float,
    latitude: float,
    radius: int = 50,
    time: Optional[List[Optional[str]]] = None,
    db_host: str = DB_HOST,
    db_port: int = DB_PORT,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
    db_name: str = DB_NAME,
    db_table: str = DB_TABLE
) -> Dict[str, Any]:
    """
    从本地数据库查询指定区域和时间范围内的船舶信息。

    Args:
        longitude (float): 中心点经度.
        latitude (float): 中心点纬度.
        radius (int): 查询半径（公里），默认为50.
        time (Optional[List[Optional[str]]]): 时间范围，格式为 [start_time, end_time].
            - [start, None]: 查询 start_time 至今的数据.
            - [None, end]: 查询历史数据直到 end_time.
            - [start, end]: 查询 start_time 和 end_time 之间的数据.
            - None: 不进行时间过滤.
        db_host (str): 数据库主机地址.
        db_port (int): 数据库端口.
        db_user (str): 数据库用户名.
        db_password (str): 数据库密码.
        db_name (str): 数据库名称.
        db_table (str): 数据表名称.

    Returns:
        Dict[str, Any]: 包含查询结果或错误信息的字典.
    """
    connection = None
    result = {
        "success": False,
        "count": 0,
        "data": [],
        "message": ""
    }

    try:
        connection = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )

        with connection.cursor() as cursor:
            
            # --------- step1 查询船舶数据 ---------
            # Base query - convert POINT to text format for easier processing
            query = f"""
            SELECT 
                id, MMSI, IMO, ship_name, call_sign, latitude, longitude,
                ST_AsText(location) as location,
                ship_heading, ship_type, track_heading, ship_length, ship_width,
                pre_loading_port, pre_loading_time, draft, update_time,
                latest_ship_position, query_time
            FROM `{db_table}`
            """
            where_clauses = []
            params = []

            # 1. Location filtering
            # For SRID 4326 (WGS84 geographic coordinates), the order in ST_GeomFromText is:
            # POINT(latitude longitude) - NOT the usual (longitude latitude)!
            # The radius is converted from km to meters.
            radius_in_meters = radius * 1000
            where_clauses.append(f"ST_Distance_Sphere(location, ST_SRID(POINT(%s, %s), 4326)) <= %s")
            params.extend([longitude, latitude, radius_in_meters])

            # 2. Time filtering
            if time and (time[0] or time[1]):
                start_time, end_time = time
                if start_time and end_time:
                    where_clauses.append("`update_time` BETWEEN %s AND %s")
                    params.extend([start_time, end_time])
                elif start_time:
                    where_clauses.append("`update_time` >= %s")
                    params.append(start_time)
                elif end_time:
                    where_clauses.append("`update_time` <= %s")
                    params.append(end_time)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            cursor.execute(query, tuple(params))
            ship_data = cursor.fetchall()

            # Convert data types to JSON-serializable formats
            from decimal import Decimal
            from datetime import datetime, date
            
            for ship in ship_data:
                # Convert Decimal to float for latitude and longitude
                if 'latitude' in ship and ship['latitude'] is not None:
                    ship['latitude'] = float(ship['latitude'])
                if 'longitude' in ship and ship['longitude'] is not None:
                    ship['longitude'] = float(ship['longitude'])
                
                # Convert datetime to ISO format string
                if 'query_time' in ship and ship['query_time'] is not None:
                    if isinstance(ship['query_time'], (datetime, date)):
                        ship['query_time'] = ship['query_time'].isoformat()
                
                # Parse location POINT text
                if 'location' in ship and ship['location']:
                    # ST_AsText returns format 'POINT(lon lat)', we parse it to a dict
                    try:
                        # Remove 'POINT(' and ')' and split by space
                        coords_str = ship['location'].replace('POINT(', '').replace(')', '')
                        coords = coords_str.split()
                        ship['location'] = {
                            'longitude': float(coords[0]), 
                            'latitude': float(coords[1])
                        }
                    except (ValueError, IndexError, AttributeError) as e:
                        # Handle potential parsing errors
                        print(f"⚠ 解析location字段失败: {e}, 原始值: {ship['location']}")
                        ship['location'] = None

            # --------- step2 广播船舶数据 ---------
            broadcast_payload = {
                "message": "船舶数据查询完成！",
                "type": "ship",
                "data_count": len(ship_data),
                "data": ship_data
            }

            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            print("############################################################################")
            # 延迟或异步方式执行，防止阻塞主流程
            import threading
            
            def send_broadcast():
                try:
                    requests.post(
                        f"{MIDDLEWARE_URL}/broadcast_default",
                        json=broadcast_payload,
                        headers=headers,
                        timeout=timeout
                    )
                except Exception as e:
                    print(f"异步广播船舶数据时出错: {e}")

            # threading.Timer(1, send_broadcast).start()
            send_broadcast()
            print("############################################################################")

            result["success"] = True
            result["count"] = len(ship_data)
            result["data"] = ship_data[:10]
            result["message"] = f"成功查询到 {result['count']} 条船舶记录。"

    except pymysql.Error as e:
        result["message"] = f"数据库错误: {e}"
        print(f"✗ {result['message']}")
    except Exception as e:
        result["message"] = f"查询过程中发生未知错误: {e}"
        print(f"✗ {result['message']}")
    finally:
        if connection:
            connection.close()

    return ship_data[:10]


if __name__ == "__main__":
    print("🚀 启动本地船舶信息查询MCP服务...")
    print(f"工具名称: local_shipinfo_search")
    
    mcp.run(transport="sse")
