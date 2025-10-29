# 基础库
import os
import sys
import time
import json
from pprint import pprint
from token import OP
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
import mysql.connector
import pymysql
from pymysql.cursors import DictCursor
from collections import OrderedDict

# 导入LangChain相关模块用于Text2SQL功能
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from urllib.parse import quote_plus
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
# project_root = Path(__file__).parent.parent
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))
from config_manager import load_config


#==========================   # 加载环境变量和配置文件  =================================

load_dotenv()
global_config = load_config()


api_key = os.environ.get('API_KEY')
llm_model= os.environ.get('LLM_MODEL')
base_url = os.environ.get('BASE_URL')


def query_image_data(
    id: Optional[List[str]] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    extent: Optional[List[float]] = None,
    table_name: Optional[str] = None,
    database: Optional[str] = None,
    limit: int = 100,
    fetch_all: bool = False,
) -> List[Dict[str, Any]]:
    """
    根据id、时间范围和地理范围从MySQL数据库中检索历史目标图像
    
    注意：如果所有查询条件参数（id、time_start、time_end、extent、fetch_all）都为空/False，
    函数将返回空列表，不会执行任何查询。

    Args:
        id: 图像ID列表（字符串类型），可选
            例如: ["LX1-1-1_01_command_vehicle_3", "LX1-1-1_01_tank_1"]
        time_start: 开始时间（日常时间格式 "YYYY-mm-DD HH:MM:SS"），可选
            例如: "2025-09-11 11:35:29"
        time_end: 结束时间（日常时间格式 "YYYY-mm-DD HH:MM:SS"），可选
            例如: "2025-09-11 23:59:59"
        extent: 四至范围 [minX(西), minY(北), maxX(东), maxY(南)]，可选
        table_name: 表名称，默认从配置读取
        database: 数据库名称，默认从配置读取
        limit: 查询结果限制数量，默认为1000条。对所有查询模式都生效
        fetch_all: 是否获取数据表内数据（忽略其他查询条件），默认False。
            当设置为True时，将忽略 id、time_start、time_end、extent 等查询条件，
            直接从表中获取数据，但仍然受 limit 参数限制
            警告：请根据需要调整 limit 参数，避免一次性获取过多数据

    Returns:
        List[Dict]: 查询结果列表，每条记录包含：
            - id: 记录ID（字符串类型，如 "LX1-1-1_01_command_vehicle_3"）
            - timestamp: Unix时间戳（秒）
            - datetime: 可读的日期时间格式（"YYYY-mm-DD HH:MM:SS"）
            - latitude: 纬度
            - longitude: 经度
            - image_id: 图像ID
            - original_image_path: 原始图像路径
            - target_image_path: 目标图像路径

    Examples:
        # 无参数调用 - 返回空列表（不执行查询）
        results = query_image_data()  # 返回 []
        
        # 按ID列表查询（默认返回前1000条）
        results = query_image_data(
            id=["LX1-1-1_01_command_vehicle_3", "LX1-1-1_01_tank_1"]
        )

        # 指定返回数量
        results = query_image_data(
            id=["LX1-1-1_01_command_vehicle_3"],
            limit=500
        )

        # 按时间范围查询（日常时间格式）
        results = query_image_data(
            time_start="2025-09-11 11:35:29",
            time_end="2025-09-11 23:59:59"
        )

        # 按地理范围查询
        results = query_image_data(
            extent=[120.0, 37.0, 121.0, 38.0]  # [西, 北, 东, 南]
        )

        # 组合查询
        results = query_image_data(
            id=["LX1-1-1_01_command_vehicle_3"],
            time_start="2025-09-11 00:00:00",
            time_end="2025-09-11 23:59:59",
            extent=[120.0, 37.0, 121.0, 38.0],
            limit=2000
        )
        
        # 获取数据表内数据（忽略查询条件，默认返回前 1000 条）
        results = query_image_data(fetch_all=True)
        
        # 获取数据表内更多数据（自定义 limit）
        results = query_image_data(fetch_all=True, limit=5000)
    """
    # 检查是否有任何查询条件，如果都没有则直接返回空列表
    has_query_condition = (
        (id is not None and len(id) > 0) or
        time_start is not None or
        time_end is not None or
        (extent is not None and len(extent) == 4) or
        fetch_all
    )
    
    if not has_query_condition:
        print("⚠ 警告: 未提供任何查询条件，返回空列表")
        return []
    
    connection = None

    try:
        # 加载配置
        
        mysql_config = global_config.get("mysql_image", {})

        # 连接到MySQL数据库
        connection = pymysql.connect(
            host=mysql_config.get("host", "localhost"),
            port=mysql_config.get("port", 3306),
            user=mysql_config.get("user", "root"),
            password=mysql_config.get("password", "123456"),
            database=database or mysql_config.get("database", "Object_detection_db"),
            charset="utf8mb4",
            cursorclass=DictCursor,
        )

        print(f"✓ 成功连接到MySQL数据库: {database or mysql_config.get('database')} '->' {table_name or mysql_config.get('table_name')}")
        # 构建SQL查询语句
        table_name = table_name or mysql_config.get("table_name")
        where_clauses = []
        params = []

        # ID列表过滤
        # 注意:如果数据库中同一个id有多条记录,这里会返回所有匹配的记录
        # 这种情况通常发生在历史记录表中,同一个目标在不同时间点被记录多次
        if id and len(id) > 0:
            placeholders = ",".join(["%s"] * len(id))
            where_clauses.append(f"id IN ({placeholders})")
            params.extend(id)

        # 时间范围过滤（将日常时间格式转换为Unix时间戳）
        if time_start is not None:
            try:
                # 将 "YYYY-mm-DD HH:MM:SS" 格式转换为 Unix 时间戳
                dt_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S")
                timestamp_start = int(dt_start.timestamp())  # 转换为Unix时间戳
                where_clauses.append("timestamp >= %s")
                params.append(timestamp_start)
                print(f"✓ 开始时间: {time_start} -> Unix时间戳: {timestamp_start}")
            except ValueError as e:
                print(
                    f"✗ 时间格式错误 (time_start): {time_start}，应为 'YYYY-mm-DD HH:MM:SS'"
                )
                raise ValueError(f"时间格式错误: {e}")

        if time_end is not None:
            try:
                # 将 "YYYY-mm-DD HH:MM:SS" 格式转换为 Unix 时间戳
                dt_end = datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S")
                timestamp_end = int(dt_end.timestamp())
                where_clauses.append("timestamp <= %s")
                params.append(timestamp_end)
                print(f"✓ 结束时间: {time_end} -> Unix时间戳: {timestamp_end}")
            except ValueError as e:
                print(
                    f"✗ 时间格式错误 (time_end): {time_end}，应为 'YYYY-mm-DD HH:MM:SS'"
                )
                raise ValueError(f"时间格式错误: {e}")


        # 地理范围过滤
        if extent and len(extent) == 4:
            # extent格式: [minX（西）, minY（南）, maxX(东), maxY(北)]
            # 经度范围: minX <= longitude <= maxX
            # 纬度范围: minY <= latitude <= maxY
            min_longitude, min_latitude, max_longitude, max_latitude = extent
            where_clauses.append("longitude >= %s AND longitude <= %s")
            params.extend([min_longitude, max_longitude])
            where_clauses.append("latitude >= %s AND latitude <= %s")
            params.extend([min_latitude, max_latitude])

        # 构建完整的SQL语句
        sql = f"SELECT * FROM `{table_name}`"

        # fetch_all=True 时查询全表数据，但仍受 limit 限制
        if fetch_all:
            print(f"⚠ 警告: 正在获取数据表内数据（忽略查询条件，但受 limit={limit} 限制）")
            # 不添加 WHERE 条件，但添加 LIMIT
        else:
            # 正常查询模式：使用条件和limit
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            print(f"✓ 查询限制: 最多返回 {limit} 条记录")
        
        # 按时间戳降序排序
        sql += " ORDER BY timestamp DESC"
        
        # 添加 LIMIT（fetch_all 和普通查询都需要）
        sql += f" LIMIT {limit}"

        # 执行查询
        start_time = time.time()
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()

        print(
            f"✓ 从MySQL数据库查询到 {len(results)} 条记录，耗时: {time.time() - start_time:.2f} 秒"
        )

        # 将结果中的 Unix 时间戳转换为可读的日期时间格式
        for result in results:
            if "timestamp" in result and result["timestamp"] is not None:
                try:
                    # 将 Unix 时间戳转换为 UTC 时间，然后加上 8 小时时区偏移
                    from datetime import timezone, timedelta
                    utc_dt = datetime.fromtimestamp(result["timestamp"], tz=timezone.utc)
                    beijing_tz = timezone(timedelta(hours=0))
                    beijing_dt = utc_dt.astimezone(beijing_tz)

                    # 添加可读的日期时间字段
                    result["datetime"] = beijing_dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, OSError) as e:
                    print(f"✗ 时间戳转换失败: {result['timestamp']}, 错误: {e}")
                    result["datetime"] = None

        # 返回JSON格式的结果列表
        return results

    except pymysql.Error as e:
        traceback.print_exc()
        raise Exception(f"MySQL错误: {str(e)}")

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"从MySQL数据库中检索历史图像失败: {str(e)}")

    finally:
        # 关闭数据库连接
        if connection:
            connection.close()


def query_equipment_data(
    extent: Optional[List[float]] = None,
    keyRegion: Optional[str] = None,
    topic: Optional[str] = None,
    layer: Optional[str] = None,
    camp: Optional[str] = None,
    status: Optional[str] = None,
    database: Optional[str] = None,
    table_name: Optional[str] = None,
    limit: int = 100,
    fetch_all: bool = False,
) -> List[Dict[str, Any]]:
    """
    根据给定条件从MySQL数据库装备数据中检索出装备数据
    
    注意：如果所有查询条件参数（extent、keyRegion、topic、layer、camp、status、fetch_all）
    都为空/None/False，函数将返回空列表，不会执行任何查询。

    Args:
        extent: 地理范围 [minX(西), minY(南), maxX(东), maxY(北)]，可选
            例如: [115.0, 39.0, 117.0, 41.0]
        keyRegion: 重点区域筛选条件，可选
            例如: "北京"
        topic: 主题筛选条件，可选
            例如: "太空态势专题"
        layer: 图层筛选条件，可选
            例如: "space"、"ocean"、"ground"
        camp: 阵营筛选条件，可选
            例如: "红方"
        status: 状态筛选条件，可选
            例如: "可用"
        database: 数据库名称，默认从配置读取
        table_name: 表名称，默认从配置读取
        limit: 查询结果限制数量，默认为100条。对所有查询模式都生效
        fetch_all: 是否获取数据表内数据（忽略其他查询条件），默认False。
            当设置为True时，将忽略 extent、keyRegion、topic、layer、camp、status 等查询条件，
            直接从表中获取数据，但仍然受 limit 参数限制
            警告：请根据需要调整 limit 参数，避免一次性获取过多数据
    Returns:
        List[Dict]: 查询结果列表，每条记录包含：
            - id: 装备ID
            - keyArea: 重点区域
            - topic: 主题
            - layer: 图层
            - camp: 阵营
            - type: 类型
            - longitude: 经度
            - latitude: 纬度
            - height: 高度
            - ISL: ISL信息
            - Status: 状态

    Examples:
        # 无参数调用 - 返回空列表（不执行查询）
        results = query_equipment_data()  # 返回 []

        # 按地理范围查询（默认返回前100条）
        results = query_equipment_data(
            extent=[115.0, 39.0, 117.0, 41.0]  # [西, 南, 东, 北]
        )

        # 按区域查询，指定返回数量
        results = query_equipment_data(keyRegion="霍尔木兹海峡", limit=50)

        # 按状态查询
        results = query_equipment_data(status="在线")

        # 组合查询
        results = query_equipment_data(
            extent=[115.0, 39.0, 117.0, 41.0],
            keyRegion="霍尔木兹海峡",
            camp="红方",
            status="可用",
            limit=200
        )
        
        # 获取数据表内数据（忽略查询条件，默认返回前 100 条）
        results = query_equipment_data(fetch_all=True)
        
        # 获取数据表内更多数据（自定义 limit）
        results = query_equipment_data(fetch_all=True, limit=5000)
    """
    # 检查是否有任何查询条件，如果都没有则直接返回空列表
    has_query_condition = (
        (extent is not None and len(extent) == 4) or
        keyRegion is not None or
        topic is not None or
        layer is not None or
        camp is not None or
        status is not None or
        fetch_all
    )
    
    if not has_query_condition:
        print("⚠ 警告: 未提供任何查询条件，返回空列表")
        return []
    
    connection = None

    try:
        # 加载配置
        mysql_config = global_config.get("mysql_equipment", {})

        database_name = database or mysql_config.get("database", "Equipments_db")
        table_name = table_name or mysql_config.get("table_name", "equipment_data")

        # 连接到MySQL数据库
        connection = pymysql.connect(
            host=mysql_config.get("host", "localhost"),
            port=mysql_config.get("port", 3306),
            user=mysql_config.get("user", "root"),
            password=mysql_config.get("password", "123456"),
            database=database_name,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )


        print(f"✓ 成功连接到MySQL数据库 -> 数据库：{database_name} + 数据表 {table_name}")

        # 构建SQL查询语句
        where_clauses = []
        params = []

        # 地理范围过滤
        if extent and len(extent) == 4:
            # extent格式: [minX（西）, minY（南）, maxX(东), maxY(北)]
            # 经度范围: minX <= longitude <= maxX
            # 纬度范围: minY <= latitude <= maxY
            min_longitude, min_latitude, max_longitude, max_latitude = extent
            where_clauses.append("longitude >= %s AND longitude <= %s")
            params.extend([min_longitude, max_longitude])
            where_clauses.append("latitude >= %s AND latitude <= %s")
            params.extend([min_latitude, max_latitude])
            print(
                f"✓ 地理范围过滤: 经度[{min_longitude}, {max_longitude}], 纬度[{min_latitude}, {max_latitude}]"
            )

        # region 筛选
        if keyRegion is not None:
            where_clauses.append("keyRegion = %s")
            params.append(keyRegion)
            print(f"✓ 区域筛选: {keyRegion}")

        # topic 筛选
        if topic is not None:
            where_clauses.append("topic = %s")
            params.append(topic)
            print(f"✓ 主题筛选: {topic}")

        # layer 筛选
        if layer is not None:
            where_clauses.append("layer = %s")
            params.append(layer)
            print(f"✓ 图层筛选: {layer}")

        # camp 筛选
        if camp is not None:
            where_clauses.append("camp = %s")
            params.append(camp)
            print(f"✓ 阵营筛选: {camp}")

        # status 筛选
        if status is not None:
            where_clauses.append("Status = %s")
            params.append(status)
            print(f"✓ 状态筛选: {status}")

        # 构建完整的SQL语句
        sql = f"SELECT * FROM `{table_name}`"

        # fetch_all=True 时查询全表数据，但仍受 limit 限制
        if fetch_all:
            print(f"⚠ 警告: 正在获取数据表内数据（忽略查询条件，但受 limit={limit} 限制）")
            # 不添加 WHERE 条件，但添加 LIMIT
        else:
            # 正常查询模式：使用条件和limit
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            print(f"✓ 查询限制: 最多返回 {limit} 条记录")

        # 按id排序
        sql += " ORDER BY id ASC"
        
        # 添加 LIMIT（fetch_all 和普通查询都需要）
        sql += f" LIMIT {limit}"

        # 执行查询
        start_time = time.time()
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()

        print(
            f"✓ 从MySQL数据库查询到 {len(results)} 条装备记录，耗时: {time.time() - start_time:.2f} 秒"
        )

        # 返回JSON格式的结果列表
        return results

    except pymysql.Error as e:
        traceback.print_exc()
        raise Exception(f"MySQL错误: {str(e)}")

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"从MySQL数据库中检索装备数据失败: {str(e)}")

    finally:
        # 关闭数据库连接
        if connection:
            connection.close()


# ====================================================== text2SQL工具   =================================================

# 全局变量，用于缓存表名列表（延迟初始化）
_tables_cache = None

# 表名列表（延迟加载，避免模块导入时连接数据库）
def get_table_names(
    database: str,  # 数据库名称
    host: str = "localhost",
    user: str = "root",
    password: str = "123456",
    port: int  = 3306,
):
    """
    获取数据库表名列表（带缓存和容错）
    
    Args:
        host: 数据库主机地址
        user: 数据库用户名
        password: 数据库密码
        database: 数据库名称
        port: 数据库端口
    
    Returns:
        list: 表名列表，如果连接失败返回空列表
    """
    global _tables_cache
    
    # 如果已缓存，直接返回
    if _tables_cache is not None:
        return _tables_cache
    
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (database,))
        _tables_cache = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        print(f"✓ 成功获取数据库表名列表: {len(_tables_cache)} 个表")
        return _tables_cache
    except Exception as e:
        print(f"⚠ 警告: 无法连接到 Text2SQL 数据库 ({host}:{port}/{database})")
        print(f"  错误信息: {str(e)}")
        print(f"  Text2SQL 功能将不可用，其他功能不受影响")
        _tables_cache = []  # 设置为空列表，避免重复尝试连接
        return _tables_cache


def get_column_order(cursor, table):
    """
    获取表的字段顺序
    
    Args:
        cursor: MySQL游标对象
        table: 表名
    
    Returns:
        list: 字段名列表
    """
    cursor.execute(f"SHOW COLUMNS FROM {table}")
    return [column[0] for column in cursor.fetchall()]


def get_table_field_info(
    table_name: str,
    database: str,  # 数据库名称
    host: str = "localhost",
    user: str = "root",
    password: str = "123456",
    port: int  = 3306,

) -> dict:
    """
    获取指定数据库表的详细字段信息
    
    Args:
        table_name: 表名
    
    Returns:
        dict: 包含表名和字段信息的字典
        {
            "table_name": "表名",
            "fields": [
                {
                    "field_name": "字段名",
                    "data_type": "数据类型",
                    "is_nullable": "是否可空",
                    "key": "键类型(PRI/UNI/MUL)",
                    "default": "默认值",
                    "extra": "额外信息(如auto_increment)"
                },
                ...
            ],
            "field_count": 字段数量
        }
    """
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        cursor = conn.cursor(dictionary=True)
        
        # 使用 SHOW COLUMNS 获取字段信息
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        
        # 格式化字段信息
        field_info_list = []
        for col in columns:
            field_info = {
                "field_name": col.get("Field", ""),
                "data_type": col.get("Type", ""),
                "is_nullable": col.get("Null", ""),
                "key": col.get("Key", ""),
                "default": col.get("Default", ""),
                "extra": col.get("Extra", "")
            }
            field_info_list.append(field_info)
        
        cursor.close()
        conn.close()
        
        return {
            "table_name": table_name,
            "fields": field_info_list,
            "field_count": len(field_info_list)
        }
        
    except mysql.connector.Error as e:
        print(f"获取表 {table_name} 字段信息失败: {str(e)}")
        return {
            "error": f"获取表字段信息失败: {str(e)}",
            "table_name": table_name
        }


def get_all_tables_field_info(
    database: str,  # 数据库名称
    host: str = "localhost",
    user: str = "root",
    password: str = "123456",
    port: int  = 3306,
) -> dict:
    """
    获取数据库中所有表的字段信息
    
    Returns:
        dict: 包含所有表字段信息的字典
        {
            "database": "数据库名",
            "table_count": 表数量,
            "tables": {
                "表名1": {字段信息},
                "表名2": {字段信息},
                ...
            }
        }
    """
    try:
        table_names = get_table_names(
            database=database,
            host=host, 
            user=user, 
            password=password, 
            port=port
        )

        tables_info = {}
        
        for table_name in table_names:
            tables_info[table_name] = get_table_field_info(
                table_name, 
                database=database, 
                host=host,
                user=user, 
                password=password, 
                port=port
            )
        
        return {
            "database":database,
            "table_count": len(table_names),
            "tables": tables_info
        }
        
    except Exception as e:
        print(f"获取所有表字段信息失败: {str(e)}")
        return {
            "error": f"获取所有表字段信息失败: {str(e)}",
            "database": database
        }


def get_field_names_only(
    table_name: str,
    database: str,  # 数据库名称
    host: str = "localhost",
    user: str = "root",
    password: str = "123456",
    port: int  = 3306,
) -> list:
    """
    仅获取表的字段名列表（简化版）
    
    Args:
        table_name: 表名
    
    Returns:
        list: 字段名列表
    """
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        cursor = conn.cursor()
        
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        field_names = [column[0] for column in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return field_names
        
    except mysql.connector.Error as e:
        print(f"获取表 {table_name} 字段名失败: {str(e)}")
        return []


# 初始化Text2SQL组件
def init_text2sql(
    database: str,  # 数据库名称
    host: str = "localhost",
    user: str = "root",
    password: str = "123456",
    port: int  = 3306,
):
    """
    初始化Text2SQL功能（延迟初始化）
    
    Returns:
        SQLDatabaseChain: SQL查询链对象，如果初始化失败返回 None
    """
    # 获取表名列表（延迟加载）
    tables = get_table_names(database=database, host=host, user=user, password=password, port=port)
    
    # 如果表名列表为空，说明数据库连接失败
    if not tables:
        print("⚠ Text2SQL 初始化失败: 无法获取数据库表名")
        return None
    
    try:
        # 构建数据库连接URI
        encoded_password = quote_plus(password)
        database_uri = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}"
        
        # 创建SQLDatabase对象
        db = SQLDatabase.from_uri(
            database_uri,  # 第一个位置参数，不需要参数名
            include_tables=tables,
            sample_rows_in_table_info=20
        )
    
        # 初始化大模型
        llm = ChatOpenAI(
            model=llm_model,  # 指定使用的模型名称
            api_key=api_key,  # API密钥
            base_url=base_url,  # API基础URL
            temperature=0,
            max_tokens=2000
        )
        
        # 自定义SQL生成提示模板
        custom_prompt = PromptTemplate(
            input_variables=["input", "table_info"],
            template=
                """
                你是一个专业的MySQL SQL助手。请基于以下数据库表结构，将用户的自然语言查询转换为MySQL兼容的SQL查询语句。

                **重要说明**:
                - 只返回SQL查询语句，不要包含任何其他文本、解释或Markdown代码块标记(如```sql或```)。
                - 不要添加任何注释或额外的格式化。
                
                数据库表结构信息:{table_info}

                用户查询: {input}

                SQL查询:
                """
        )
        
        # 创建SQL查询链
        sql_chain = SQLDatabaseChain.from_llm(
            llm,
            db,
            prompt=custom_prompt,
            verbose=True,
            return_intermediate_steps=True
        )
        
        print("✓ Text2SQL 初始化成功")
        return sql_chain

    except Exception as e:
        print(f"⚠ Text2SQL 初始化失败: {str(e)}")
        return None


if __name__ == "__main__":


    # 初始化数据库连接
    sql_chain = init_text2sql(
        database="Equipments_db",
        host="localhost",
        user="root",
        password="123456",
        port=3306
    )

    if sql_chain is None:
        print("⚠ Text2SQL 初始化失败")
        exit(1)

    print("\n" + "="*80 + "\n")

    # 测试装备数据查询
    print("💯 ===   测试MySQL装备数据查询  ===")

    # 测试1: 查询所有装备
    print("\n【测试1】查询所有装备数据:")
    equipment_results = query_equipment_data()
    print(f"查询到 {len(equipment_results)} 条装备数据")
    if equipment_results:
        pprint(equipment_results[:2])  # 只显示前2条

    # 测试2: 按地理范围查询
    print("\n【测试2】按地理范围查询:")
    equipment_results = query_equipment_data(
        extent=[80.0, 10.0, 150.0, 50.0]  # 北京地区范围
    )
    print(f"查询到 {len(equipment_results)} 条装备数据")
    if equipment_results:
        pprint(equipment_results[:2])

    # 测试3: 按状态查询
    print("\n【测试3】按状态查询:")
    equipment_results = query_equipment_data(status="可用")
    print(f"查询到 {len(equipment_results)} 条装备数据")
    if equipment_results:
        pprint(equipment_results[:2])

    # 测试4: 组合查询
    print("\n【测试4】组合查询（区域+阵营+状态）:")
    equipment_results = query_equipment_data(
        region="霍尔木兹海峡",
        camp="红方",
        status="可用"
    )
    print(f"查询到 {len(equipment_results)} 条装备数据")
    if equipment_results:
        pprint(equipment_results[:2])








