"""
船舶信息查询工具 - 支持船舶数据查询和本地化存储

功能:
1. 根据经纬度坐标查询船舶实时信息
2. 将查询到的船舶数据保存到MySQL数据库（支持空间索引）
3. 支持船舶数据的自动去重和更新
4. 使用MySQL空间函数进行高效的地理位置查询

主要函数:
- get_ship_numbers_by_location: 根据位置获取船舶MMSI和IMO列表
- get_ships_info_by_imo_list: 根据IMO列表获取船舶详细信息
- save_ship_data_to_mysql: 将船舶数据保存到MySQL数据库（自动创建空间索引）
- shipinfo_search_tool: 主逻辑函数，查询并保存船舶数据
- query_ships_by_radius: 使用空间索引查询指定半径内的船舶
- query_ships_in_bounding_box: 使用空间索引查询矩形区域内的船舶

数据库表结构:
- 数据库: ship_info_db
- 表名: ship_realtime_data
- 字段: MMSI, IMO, ship_name, call_sign, latitude, longitude, location(POINT),
        ship_heading, ship_type, track_heading, ship_length, ship_width, 
        pre_loading_port, pre_loading_time, draft, update_time, 
        latest_ship_position, query_time
- 空间索引: idx_location (基于location字段，SRID 4326)

空间查询示例:
    # 查询巴生港50公里内的船舶
    ships = query_ships_by_radius(
        center_longitude=101.4,
        center_latitude=3.0,
        radius_km=50
    )
    
    # 查询南海区域内的船舶
    ships = query_ships_in_bounding_box(
        min_longitude=110.0, min_latitude=3.0,
        max_longitude=118.0, max_latitude=21.0
    )

详细使用说明请参考: utils/SPATIAL_INDEX_USAGE.md
"""

from pprint import pprint
from typing import Any, Dict, List, Union
import requests
import json
import urllib.parse
import re
from fastmcp import FastMCP
import os

# ===================== 服务配置信息 =====================
# 优先获取环境变量，若无则使用默认值
from dotenv import load_dotenv
load_dotenv()
UAVIMG_SERVER_URL = os.getenv("UAVIMG_SERVER_URL", "http://localhost:5000")# 无人机及图像服务API
# UAVIMG_SERVER_URL = os.getenv("UAVIMG_SERVER_URL", "http://192.168.71.232:5000")# 无人机及图像服务API

timeout = 30
# 创建MCP服务器实例
mcp = FastMCP(name="船舶信息查询工具", port=8202)

def get_ship_type(type_code):
    """根据船舶类型代码返回船舶类型名称"""
    ship_type_mapping = {
        55: "执法船",
        60: "客船", 
        70: "货船",
        71: "货船",
        80: "油轮"
    }
    
    try:
        code = int(type_code) if type_code != "null" and type_code is not None else None
        return ship_type_mapping.get(code, "未知")
    except (ValueError, TypeError):
        return "未知"


def convert_dms_to_decimal(coordinate_str):
    """
    将度分格式的经纬度转换为十进制度数
    
    参数:
        coordinate_str (str): 度分格式的坐标字符串，例如：
                             "N 20度12.3322分" 或 "E 110度8.0286分"
    
    返回:
        float: 十进制度数，经度范围[-180, 180]，纬度范围[-90, 90]
               如果解析失败返回None
    """
    if not coordinate_str or coordinate_str == "null":
        return None
    
    try:
        # 移除多余的空格并转换为字符串
        coord_str = str(coordinate_str).strip()
        
        # 提取方向（N/S/E/W）
        direction = None
        if coord_str.startswith(('N', 'S', 'E', 'W')):
            direction = coord_str[0]
            coord_str = coord_str[1:].strip()
        elif coord_str.endswith(('N', 'S', 'E', 'W')):
            direction = coord_str[-1]
            coord_str = coord_str[:-1].strip()
        
        # 使用正则表达式提取度和分
        import re
        pattern = r'(\d+(?:\.\d+)?)度(\d+(?:\.\d+)?)分'
        match = re.search(pattern, coord_str)
        
        if match:
            degrees = float(match.group(1))
            minutes = float(match.group(2))
            
            # 转换为十进制度数
            decimal_degrees = degrees + minutes / 60.0
            
            # 根据方向确定正负号
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            
            # 检查范围
            if direction in ['N', 'S']:  # 纬度
                if decimal_degrees < -90 or decimal_degrees > 90:
                    return None
            elif direction in ['E', 'W']:  # 经度
                if decimal_degrees < -180 or decimal_degrees > 180:
                    return None
            
            return round(decimal_degrees, 8)
        else:
            # 如果不是度分格式，尝试直接转换为浮点数
            try:
                return round(float(coord_str), 8)
            except ValueError:
                return None
                
    except Exception as e:
        print(f"坐标转换错误: {e}")
        return None


def parse_ship_info(data_list):
    """解析船舶信息数组并转换为字典格式"""
    if not data_list or len(data_list) < 18:
        return None
    # print(f"数据长度: {len(data_list)}************************************************************************************")
    # print(data_list)
    
    # 转换经纬度格式
    latitude_decimal = convert_dms_to_decimal(data_list[4])
    longitude_decimal = convert_dms_to_decimal(data_list[6])
    
    ship_info = {
        "ship_name": data_list[0] if data_list[0] is not None else "未知",  # 船名
        "MMSI": data_list[1] if data_list[1] is not None else "未知",  # 船舶识别号
        "IMO": data_list[2] if data_list[2] is not None else "未知",  # 国际海事组织号
        "call_sign": data_list[3] if data_list[3] is not None else "未知",  # 呼号
        "latitude": latitude_decimal if latitude_decimal is not None else "未知",  # 纬度
        "longitude": longitude_decimal if longitude_decimal is not None else "未知",  # 经度
        "ship_heading": f"{data_list[7]}度" if data_list[7] != "null" else "未知", # 船首方向
        "ship_type": get_ship_type(data_list[8]) if len(data_list) > 8 else "未知", # 船舶类型
        "track_heading": f"{data_list[9]}度" if data_list[9] != "null" else "未知", # 航迹方向
        "ship_length": f"{data_list[12]}米" if data_list[12] != "null" else "未知", # 船长度
        "pre_loading_port": data_list[13] if data_list[13] is not None else "未知", # 预到港
        "ship_width": f"{data_list[14]}米" if data_list[14] != "null" else "未知", # 船宽度
        "pre_loading_time": data_list[15] if data_list[15] is not None else "未知", # 预到港时间
        "draft": f"{data_list[16]}米" if data_list[16] != "null" else "未知", # 吃水深度
        "update_time": data_list[17] if data_list[17] is not None else "未知", # 更新时间
    }
    
    # 添加最新船位信息（如果存在）
    if len(data_list) > 25 and data_list[25] != "null":
        ship_info["latest_ship_position"] = data_list[25] if data_list[25] is not None else "未知" # 最新船位信息
    
    return ship_info


def extract_ship_numbers(response_text):
    """
    从返回的callback数据中提取船编号，分别返回MMSI和IMO列表
    """
    mmsi_list = []
    imo_list = []
    
    try:
        # 使用正则表达式提取callback中的数据部分
        match = re.search(r'callback\((.*)\)', response_text)
        if match:
            data_str = match.group(1)
            # 将JavaScript的null替换为Python的None，然后使用json.loads解析
            data_str = data_str.replace('null', 'None')
            # 使用eval解析数据（已处理null值问题）
            ship_data = eval(data_str)
            
            # 遍历每艘船的数据
            for ship in ship_data:
                if len(ship) > 18:  # 确保数据长度足够
                    mmsi = ship[6]  # MMSI号在索引6
                    imo = ship[18]  # IMO号在索引18
                    
                    # 添加MMSI号（如果不为空且不为None）
                    if mmsi and mmsi != "0" and mmsi != "" and mmsi is not None:
                        mmsi_list.append(mmsi)
                    
                    # 添加IMO号（如果不为空且不为None）
                    if imo and imo != "0" and imo != "" and imo is not None:
                        imo_list.append(imo)
    
    except Exception as e:
        print(f"解析船编号时出错: {e}")
    
    return mmsi_list, imo_list

# 76.43702828517625
def get_ship_numbers_by_location(center_x, center_y, resolution='10.43702828517625', verbose=False):
    """
    根据地点坐标获取船舶编号列表
    
    参数:
        center_x (str): 中心点X坐标（经度）
        center_y (str): 中心点Y坐标（纬度）
        resolution (str): 分辨率，默认为'10.43702828517625'
        verbose (bool): 是否打印详细信息，默认为False
    
    返回:
        tuple: (mmsi_list, imo_list) - MMSI列表和IMO列表
    """
    url = "https://www.chinaports.com/shiptracker/shipinit.do"
    
    payload = {
        'method': 'poszoom',
        'center_x': str(center_x),
        'center_y': str(center_y),
        'resolution': str(resolution),
        'param1': 'true',
        'pos': '1',
        'type': '0'
    }
    
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Accept': '*/*',
        'Host': 'www.chinaports.com',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Cookie': 'JSESSIONID=F150860A0DBD03F7B564690D25B2F262'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        
        if verbose:
            print(f"状态码: {response.status_code}")
            print(f"响应长度: {len(response.text)}")
        
        if response.status_code == 200:
            # 提取船编号列表
            mmsi_list, imo_list = extract_ship_numbers(response.text)
            return mmsi_list, imo_list
        else:
            if verbose:
                print("错误响应:")
                print(response.text)
            return [], []
            
    except Exception as e:
        if verbose:
            print(f"请求异常: {e}")
        return [], []


def get_ships_info_by_imo_list(imo_list, verbose=False):
    """
    根据IMO列表获取船舶详细信息
    
    参数:
        imo_list: IMO号码列表
        verbose: 是否显示详细日志信息
    
    返回:
        船舶信息列表，每个元素包含船舶的详细信息字典
    """
    ships_info = []
    url = "https://www.chinaports.com/shiptracker/shipinit.do"
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    for imo in imo_list:
        if verbose:
            print(f"\n正在查询IMO: {imo}")
        
        payload = {
            'method': 'shipInfo',
            'userid': str(imo),  # userid参数实际上是IMO编号
            'source': '0',
            'num': '1759219652598'
        }
        
        try:
            response = requests.post(url, headers=headers, data=payload)
            
            if response.status_code == 200:
                try:
                    response_data = json.loads(response.text)
                    if isinstance(response_data, list):
                        ship_dict = parse_ship_info(response_data)
                        if ship_dict:
                            ships_info.append(ship_dict)
                            if verbose:
                                print(f"成功获取船舶信息: {ship_dict.get('船名', 'Unknown')}")
                        else:
                            if verbose:
                                print(f"IMO {imo}: 数据格式不正确或数据不完整")
                    else:
                        if verbose:
                            print(f"IMO {imo}: 响应数据格式异常")
                except json.JSONDecodeError:
                    # 尝试eval解析
                    response_text = response.text.strip()
                    if response_text.startswith('[') and response_text.endswith(']'):
                        try:
                            # 处理null值
                            response_text = response_text.replace('null', 'None')
                            data_list = eval(response_text)
                            ship_dict = parse_ship_info(data_list)
                            if ship_dict:
                                ships_info.append(ship_dict)
                                if verbose:
                                    print(f"成功获取船舶信息: {ship_dict.get('船名', 'Unknown')}")
                        except Exception as eval_error:
                            if verbose:
                                print(f"IMO {imo}: 数组解析失败: {eval_error}")
            else:
                if verbose:
                    print(f"IMO {imo}: HTTP错误 {response.status_code}")
                    
        except Exception as e:
            if verbose:
                print(f"IMO {imo}: 请求异常: {e}")
    
    return ships_info


def save_ship_data_to_mysql(
    # 船舶数据参数
    ship_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    # MySQL存储参数
    host: str = "localhost",
    port: int = 3306,
    user: str = "root", 
    password: str = "123456",
    database: str = "ship_info_db",
    table_name: str = "ship_realtime_data",
    auto_create_db: bool = True,
    auto_create_table: bool = True,
) -> Dict[str, Any]:
    """
    将船舶数据写入MySQL数据库

    触发关键词: 保存、存储、写入、数据库、MySQL

    Args:
        ship_data: 船舶数据，可以是单个字典或字典列表（从get_ships_info_by_imo_list返回的数据）
        host: MySQL服务器地址，默认localhost
        port: MySQL端口，默认3306
        user: MySQL用户名，默认root
        password: MySQL密码，默认123456
        database: 数据库名称，默认ship_info_db
        table_name: 表名称，默认ship_realtime_data
        auto_create_db: 是否自动创建数据库，默认True
        auto_create_table: 是否自动创建表，默认True

    Returns:
        Dict[str, Any]: 包含执行结果的字典
            - success: bool, 是否成功
            - inserted_count: int, 插入的记录数
            - updated_count: int, 更新的记录数
            - message: str, 执行消息
            - errors: List[str], 错误信息列表

    Examples:
        # 保存船舶数据
        result = save_ship_data_to_mysql(
            ship_data=ships_info_list
        )
    """
    
    import pymysql
    from pymysql.cursors import DictCursor
    from datetime import datetime
    
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
        if isinstance(ship_data, dict):
            data_list = [ship_data]
        elif isinstance(ship_data, list):
            data_list = ship_data
        else:
            raise ValueError(f"不支持的数据类型: {type(ship_data)}")

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
                `id` INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
                `MMSI` VARCHAR(20) NULL COMMENT '船舶识别号',
                `IMO` VARCHAR(20) NULL COMMENT '国际海事组织号',
                `ship_name` VARCHAR(100) NULL COMMENT '船名',
                `call_sign` VARCHAR(20) NULL COMMENT '呼号',
                `latitude` DECIMAL(10, 8) NULL COMMENT '纬度',
                `longitude` DECIMAL(11, 8) NULL COMMENT '经度',
                `location` POINT NOT NULL SRID 4326 COMMENT '地理位置点（用于空间索引）',
                `ship_heading` VARCHAR(20) NULL COMMENT '船首方向',
                `ship_type` VARCHAR(50) NULL COMMENT '船舶类型',
                `track_heading` VARCHAR(20) NULL COMMENT '航迹方向',
                `ship_length` VARCHAR(20) NULL COMMENT '船长度',
                `ship_width` VARCHAR(20) NULL COMMENT '船宽度',
                `pre_loading_port` VARCHAR(100) NULL COMMENT '预到港',
                `pre_loading_time` VARCHAR(50) NULL COMMENT '预到港时间',
                `draft` VARCHAR(20) NULL COMMENT '吃水深度',
                `update_time` VARCHAR(50) NULL COMMENT '更新时间',
                `latest_ship_position` VARCHAR(200) NULL COMMENT '最新船位信息',
                `query_time` DATETIME NULL COMMENT '查询时间',
                PRIMARY KEY (`id`),
                UNIQUE KEY `idx_mmsi_update` (`MMSI`, `update_time`),
                INDEX `idx_ship_name` (`ship_name`),
                INDEX `idx_ship_type` (`ship_type`),
                INDEX `idx_update_time` (`update_time`),
                SPATIAL INDEX `idx_location` (`location`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='船舶实时数据表';
            """

            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                connection.commit()
                print(f"✓ 数据表 '{table_name}' 已确认存在")
                
                # 检查并添加location列（如果表已存在但没有该列）
                check_column_sql = f"""
                SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{database}' 
                AND TABLE_NAME = '{table_name}' 
                AND COLUMN_NAME = 'location'
                """
                cursor.execute(check_column_sql)
                result_check = cursor.fetchone()
                
                if result_check['count'] == 0:
                    # 添加location列（先允许NULL，填充后再转为NOT NULL）
                    alter_add_column_sql = f"""
                    ALTER TABLE `{table_name}` 
                    ADD COLUMN `location` POINT SRID 4326 NULL COMMENT '地理位置点（用于空间索引）' AFTER `longitude`
                    """
                    cursor.execute(alter_add_column_sql)
                    print(f"✓ 已为表 '{table_name}' 添加 location 列")

                # 尝试用已有经纬度填充location列
                fill_location_sql = f"""
                UPDATE `{table_name}`
                SET `location` = ST_SRID(POINT(`longitude`, `latitude`), 4326)
                WHERE `location` IS NULL
                  AND `longitude` IS NOT NULL
                  AND `latitude` IS NOT NULL
                  AND ABS(`latitude`) <= 90
                  AND ABS(`longitude`) <= 180
                """
                cursor.execute(fill_location_sql)

                # 检查location列中是否仍然存在NULL
                cursor.execute(f"SELECT COUNT(*) AS count FROM `{table_name}` WHERE `location` IS NULL")
                null_count = cursor.fetchone()["count"]

                if null_count == 0:
                    # 将location列改为NOT NULL并创建空间索引
                    modify_location_sql = f"""
                    ALTER TABLE `{table_name}`
                    MODIFY COLUMN `location` POINT NOT NULL SRID 4326 COMMENT '地理位置点（用于空间索引）'
                    """
                    cursor.execute(modify_location_sql)
                    print(f"✓ 已将表 '{table_name}' 的 location 列设置为 NOT NULL")

                    # 检查是否已经存在空间索引
                    check_index_sql = f"""
                    SELECT COUNT(*) AS count FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE TABLE_SCHEMA = '{database}'
                    AND TABLE_NAME = '{table_name}'
                    AND INDEX_NAME = 'idx_location'
                    """
                    cursor.execute(check_index_sql)
                    index_exists = cursor.fetchone()["count"] > 0

                    if not index_exists:
                        try:
                            alter_add_spatial_index_sql = f"""
                            ALTER TABLE `{table_name}` 
                            ADD SPATIAL INDEX `idx_location` (`location`)
                            """
                            cursor.execute(alter_add_spatial_index_sql)
                            print(f"✓ 已为表 '{table_name}' 添加空间索引")
                        except pymysql.Error as e:
                            if "Duplicate key name" not in str(e):
                                print(f"⚠ 添加空间索引时出现警告: {e}")
                else:
                    print(
                        "⚠ 检测到仍有记录缺少经纬度信息，未将 location 列设置为 NOT NULL，空间索引暂未创建。"
                    )

                connection.commit()

        # 准备插入/更新数据
        with connection.cursor() as cursor:
            for idx, item in enumerate(data_list):
                try:
                    # 处理经纬度，如果是字符串"未知"则设为None
                    latitude = _normalize_coordinate(item.get("latitude"))
                    longitude = _normalize_coordinate(item.get("longitude"))

                    if latitude is None or longitude is None:
                        print(
                            f"⚠ 第{idx+1}条数据缺少有效经纬度，已跳过 (船名: {item.get('ship_name', 'unknown')})"
                        )
                        continue

                    # 如果纬度超出范围而经度在合理范围内，尝试自动对调
                    if abs(latitude) > 90 and abs(longitude) <= 90:
                        latitude, longitude = longitude, latitude

                    # 如果经度超出范围且纬度在合理范围内，同样对调
                    if abs(longitude) > 180 and abs(latitude) <= 90:
                        latitude, longitude = longitude, latitude

                    if abs(latitude) > 90 or abs(longitude) > 180:
                        print(
                            f"⚠ 第{idx+1}条数据的经纬度超出有效范围，已跳过 (船名: {item.get('ship_name', 'unknown')})"
                        )
                        continue

                    latitude = round(float(latitude), 8)
                    longitude = round(float(longitude), 8)

                    # 使用INSERT ... ON DUPLICATE KEY UPDATE语法实现upsert，并构造地理位置点
                    insert_sql = f"""
                    INSERT INTO `{table_name}` (
                        MMSI, IMO, ship_name, call_sign, latitude, longitude, location,
                        ship_heading, ship_type, track_heading, ship_length, ship_width,
                        pre_loading_port, pre_loading_time, draft, update_time,
                        latest_ship_position, query_time
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, ST_SRID(POINT(%s, %s), 4326),
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        IMO = VALUES(IMO),
                        ship_name = VALUES(ship_name),
                        call_sign = VALUES(call_sign),
                        latitude = VALUES(latitude),
                        longitude = VALUES(longitude),
                        location = VALUES(location),
                        ship_heading = VALUES(ship_heading),
                        ship_type = VALUES(ship_type),
                        track_heading = VALUES(track_heading),
                        ship_length = VALUES(ship_length),
                        ship_width = VALUES(ship_width),
                        pre_loading_port = VALUES(pre_loading_port),
                        pre_loading_time = VALUES(pre_loading_time),
                        draft = VALUES(draft),
                        update_time = VALUES(update_time),
                        latest_ship_position = VALUES(latest_ship_position),
                        query_time = VALUES(query_time)
                    """

                    values = (
                        item.get("MMSI") if item.get("MMSI") != "未知" else None,
                        item.get("IMO") if item.get("IMO") != "未知" else None,
                        item.get("ship_name") if item.get("ship_name") != "未知" else None,
                        item.get("call_sign") if item.get("call_sign") != "未知" else None,
                        latitude,
                        longitude,
                        longitude,
                        latitude,
                        item.get("ship_heading") if item.get("ship_heading") != "未知" else None,
                        item.get("ship_type") if item.get("ship_type") != "未知" else None,
                        item.get("track_heading") if item.get("track_heading") != "未知" else None,
                        item.get("ship_length") if item.get("ship_length") != "未知" else None,
                        item.get("ship_width") if item.get("ship_width") != "未知" else None,
                        item.get("pre_loading_port") if item.get("pre_loading_port") != "未知" else None,
                        item.get("pre_loading_time") if item.get("pre_loading_time") != "未知" else None,
                        item.get("draft") if item.get("draft") != "未知" else None,
                        item.get("update_time") if item.get("update_time") != "未知" else None,
                        item.get("latest_ship_position") if item.get("latest_ship_position") != "未知" else None,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    affected_rows = cursor.execute(insert_sql, values)

                    # 判断是插入还是更新
                    if affected_rows == 1:
                        result["inserted_count"] += 1
                    elif affected_rows == 2:
                        result["updated_count"] += 1

                except Exception as e:
                    error_msg = f"处理第{idx+1}条数据时出错 (船名: {item.get('ship_name', 'unknown')}): {str(e)}"
                    result["errors"].append(error_msg)
                    print(f"✗ {error_msg}")

            # 提交事务
            connection.commit()

        # 设置成功状态
        total_processed = result["inserted_count"] + result["updated_count"]
        result["success"] = True
        result["message"] = (
            f"成功处理 {total_processed} 条船舶数据 (插入: {result['inserted_count']}, 更新: {result['updated_count']})"
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


def _normalize_coordinate(value):
    """将输入的经纬度值转换为浮点数，如果不可用则返回None"""
    if value in (None, "未知"):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "null":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# @mcp.tool
def save_shipinfo_to_db(
    center_x, 
    center_y,
    save_to_db: bool = True,
    db_host: str = "localhost",
    db_port: int = 3306,
    db_user: str = "root",
    db_password: str = "123456",
    db_name: str = "shipinfo_db",
    db_table: str = "shipinfo_metadata"
):
    """
    shipinfo_search_tool工具用于查询某地点附近的实时船舶数据，能够根据经纬度位置获取该区域内的船舶详细信息列表，并可选地保存到本地MySQL数据库.

    可参考的数据：南海的大致范围为东经110度到118度，北纬3度到21度。马六甲海峡区域的大致范围为东经98度到104度，北纬1度到8度。巴生港大致位置为北纬3°，东经101度。

    注意：当用户要查询某个区域内的船舶信息时，可从该区域内取典型的位置点进行查询，如马六甲海峡地区，可查询：（北纬3度，东经101度）或（北纬2.2度，东经101.12度）
    
    参数:
        center_x (str/float): 中心点X坐标（经度）
        center_y (str/float): 中心点Y坐标（纬度）
        save_to_db (bool): 是否保存到数据库，默认True
        db_host (str): MySQL服务器地址，默认localhost
        db_port (int): MySQL端口，默认3306
        db_user (str): MySQL用户名，默认root
        db_password (str): MySQL密码，默认123456
        db_name (str): 数据库名称，默认ship_info_db
        db_table (str): 表名称，默认ship_realtime_data
 
    返回:
        Dict: 包含船舶信息和数据库保存结果的字典
            - ships_info: 船舶详细信息列表
            - db_result: 数据库保存结果（如果save_to_db=True）
    """
    
    print(f"🌍 开始查询位置 ({center_x}, {center_y}) 的船舶信息...")

    try:
        # 第一步：根据位置获取船舶MMSI和IMO列表
        mmsi_list, imo_list = get_ship_numbers_by_location(
            center_x=center_x, 
            center_y=center_y, 
            resolution='50.43702828517625',  # resolution (str): 分辨率，默认为'76.43702828517625'
        )
        
        print(f"📡 在该位置发现 {len(imo_list)} 艘船舶（IMO列表）")
        print(f"IMO列表: {imo_list}")
        
        if not imo_list:
            print("❌ 该位置当前没有发现任何船舶")
            return {
                "ships_info": [],
                "message": "该位置当前没有发现任何船舶"
            }
        
        # 第二步：获取船舶详细信息
        ships_info = get_ships_info_by_imo_list(imo_list, verbose=True)
        
        # print("船舶详细信息: ")
        # pprint(ships_info)

        # 第三步：保存到数据库（如果启用）
        db_result = None
        if save_to_db and ships_info:
            print(f"\n💾 开始保存船舶数据到MySQL数据库...")
            db_result = save_ship_data_to_mysql(
                ship_data=ships_info,
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
                table_name=db_table,
                auto_create_db=True,
                auto_create_table=True
            )
            print(f"💾 数据库保存结果: {db_result['message']}")

        # # 第四步：广播船舶数据到前端
        # broadcast_payload = {
        #     "message": "船舶数据查询完成！",
        #     "type": "ship",
        #     "data": ships_info,
        # }
        
        # headers = {
        #     "Content-Type": "application/json",
        #     "Accept": "application/json"
        # }

        # print("############################################################################")
        # # 延迟或异步方式执行，防止阻塞主流程
        # import threading
        # def send_broadcast():
        #     try:
        #         requests.post(
        #             f"{UAVIMG_SERVER_URL}/broadcast_default",
        #             json=broadcast_payload,
        #             headers=headers,
        #             timeout=timeout
        #         )
        #     except Exception as e:
        #         print(f"异步广播船舶数据时出错: {e}")

        # send_broadcast()
        # print("############################################################################")

        # 返回结果
        result = {
            "ships_info": ships_info[:10],  # 返回前10条供显示
            "total_ships": len(ships_info),
            "query_location": {
                "longitude": center_x,
                "latitude": center_y
            }
        }
        
        if db_result:
            result["db_result"] = db_result
        
        return result
    
    except requests.exceptions.Timeout:
        return {
            "error": f"船舶动态信息库请求超时（{timeout}秒）",
            "ships_info": []
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "船舶动态信息库的连接错误，请检查网络连接",
            "ships_info": []
        }
    except requests.exceptions.HTTPError as e:
        return {
            "error": f"船舶动态信息库的HTTP错误: {e}",
            "ships_info": []
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"船舶动态信息库请求失败: {e}",
            "ships_info": []
        }
    except json.JSONDecodeError:
        return {
            "error": "船舶动态信息库响应不是有效的JSON格式",
            "ships_info": []
        }
    except Exception as e:
        return {
            "error": f"处理船舶数据时发生错误: {e}",
            "ships_info": []
        }
  

def query_ships_by_radius(
    center_longitude: float,
    center_latitude: float,
    radius_km: float = 50.0,
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "123456",
    database: str = "shipinfo_db",
    table_name: str = "shipinfo_metadata",
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    使用MySQL空间函数查询指定半径内的船舶信息
    
    参数:
        center_longitude: 中心点经度
        center_latitude: 中心点纬度
        radius_km: 搜索半径（公里），默认50km
        host: MySQL服务器地址
        port: MySQL端口
        user: MySQL用户名
        password: MySQL密码
        database: 数据库名称
        table_name: 表名称
        limit: 最多返回结果数
    
    返回:
        船舶信息列表，包含距离字段（单位：公里）
        
    示例:
        # 查询巴生港50公里内的所有船舶
        ships = query_ships_by_radius(
            center_longitude=101.4,
            center_latitude=3.0,
            radius_km=50
        )
        
        # 打印结果
        for ship in ships:
            print(f"船名: {ship['ship_name']}, 距离: {ship['distance_km']:.2f}公里")
    """
    import pymysql
    from pymysql.cursors import DictCursor
    
    connection = None
    result_list = []
    
    try:
        # 连接数据库
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )
        
        with connection.cursor() as cursor:
            # 使用ST_Distance_Sphere计算球面距离
            # 注意：ST_Distance_Sphere返回的距离单位是米，需要除以1000转换为公里
            query_sql = f"""
            SELECT 
                id, MMSI, IMO, ship_name, call_sign,
                latitude, longitude,
                ship_heading, ship_type, track_heading,
                ship_length, ship_width, pre_loading_port,
                pre_loading_time, draft, update_time,
                latest_ship_position, query_time,
                ST_Distance_Sphere(
                    location,
                    ST_SRID(POINT(%s, %s), 4326)
                ) / 1000 AS distance_km
            FROM `{table_name}`
            WHERE location IS NOT NULL
            AND ST_Distance_Sphere(
                location,
                ST_SRID(POINT(%s, %s), 4326)
            ) <= %s
            ORDER BY distance_km ASC
            LIMIT %s
            """
            
            # 半径转换为米
            radius_m = radius_km * 1000
            
            cursor.execute(
                query_sql, 
                (center_longitude, center_latitude, center_longitude, center_latitude, radius_m, limit)
            )
            
            result_list = cursor.fetchall()
            
            # 转换distance_km为浮点数
            for item in result_list:
                if item.get('distance_km') is not None:
                    item['distance_km'] = float(item['distance_km'])
            
            print(f"✓ 在 ({center_longitude}, {center_latitude}) 的 {radius_km}km 半径内找到 {len(result_list)} 艘船舶")
            
    except pymysql.Error as e:
        print(f"✗ MySQL查询错误: {e}")
    except Exception as e:
        print(f"✗ 查询错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if connection:
            connection.close()
    
    return result_list


def query_ships_in_bounding_box(
    min_longitude: float,
    min_latitude: float,
    max_longitude: float,
    max_latitude: float,
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "123456",
    database: str = "shipinfo_db",
    table_name: str = "shipinfo_metadata",
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    使用MySQL空间函数查询矩形区域内的船舶信息
    
    参数:
        min_longitude: 最小经度（西）
        min_latitude: 最小纬度（南）
        max_longitude: 最大经度（东）
        max_latitude: 最大纬度（北）
        host: MySQL服务器地址
        port: MySQL端口
        user: MySQL用户名
        password: MySQL密码
        database: 数据库名称
        table_name: 表名称
        limit: 最多返回结果数
    
    返回:
        船舶信息列表
        
    示例:
        # 查询南海某区域内的所有船舶（东经110-118度，北纬3-21度）
        ships = query_ships_in_bounding_box(
            min_longitude=110.0,
            min_latitude=3.0,
            max_longitude=118.0,
            max_latitude=21.0
        )
    """
    import pymysql
    from pymysql.cursors import DictCursor
    
    connection = None
    result_list = []
    
    try:
        # 连接数据库
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )
        
        with connection.cursor() as cursor:
            # 使用ST_Within和ST_GeomFromText进行矩形区域查询
            # SRID 4326 要求顺序为 POINT(纬度, 经度)
            query_sql = f"""
            SELECT 
                id, MMSI, IMO, ship_name, call_sign,
                latitude, longitude,
                ship_heading, ship_type, track_heading,
                ship_length, ship_width, pre_loading_port,
                pre_loading_time, draft, update_time,
                latest_ship_position, query_time
            FROM `{table_name}`
            WHERE location IS NOT NULL
            AND ST_Within(
                location,
                ST_GeomFromText('POLYGON((%s %s, %s %s, %s %s, %s %s, %s %s))', 4326)
            )
            ORDER BY update_time DESC
            LIMIT %s
            """
            
            # 构建矩形的5个点（闭合）
            # 注意：SRID 4326 中坐标顺序为 (纬度, 经度)
            cursor.execute(
                query_sql,
                (
                    min_latitude, min_longitude,  # 左下 (纬度, 经度)
                    min_latitude, max_longitude,  # 右下 (纬度, 经度)
                    max_latitude, max_longitude,  # 右上 (纬度, 经度)
                    max_latitude, min_longitude,  # 左上 (纬度, 经度)
                    min_latitude, min_longitude,  # 闭合到左下 (纬度, 经度)
                    limit
                )
            )
            
            result_list = cursor.fetchall()
            
            print(f"✓ 在矩形区域 [({min_longitude}, {min_latitude}) - ({max_longitude}, {max_latitude})] 内找到 {len(result_list)} 艘船舶")
            
    except pymysql.Error as e:
        print(f"✗ MySQL查询错误: {e}")
    except Exception as e:
        print(f"✗ 查询错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if connection:
            connection.close()
    
    return result_list


if __name__ == "__main__":
    # mcp.run(transport='sse')

    # 示例1：查询并保存船舶数据
    # save_shipinfo_to_db(center_x=104.764948, center_y=28.792433)  //
    # save_shipinfo_to_db(center_x=121.63, center_y=24.0)         //花莲港附近
    # save_shipinfo_to_db(center_x=121.24, center_y=25.11)         //桃园机场附近
    #save_shipinfo_to_db(center_x=121.76, center_y=25.16)         #//滨海公园附近
    # save_shipinfo_to_db(center_x=120.499, center_y=24.277)         #//台中公园附近24.277787715712975, 120.49902044996074
    # save_shipinfo_to_db(center_x=120.2748, center_y=22.599)         #//高雄公园附近


    save_shipinfo_to_db(center_x=118.04656, center_y=24.45498)
    # 24.454983443084515, 118.04656345803896  厦门嵩屿
    # 示例2：使用空间索引查询指定半径内的船舶
    # ships = query_ships_by_radius(
    #     center_longitude=104.764948,
    #     center_latitude=28.792433,
    #     radius_km=10
    # )
    # # # print(f"找到 {len(ships)} 艘船舶")
    # for ship in ships[:5]:  # 打印前5条
    #     print(f"  - {ship['ship_name']}: {ship['distance_km']:.2f}km")
    
    # 示例3：查询矩形区域内的船舶
    # ships = query_ships_in_bounding_box(
    #     min_longitude=110.0,
    #     min_latitude=3.0,
    #     max_longitude=118.0,
    #     max_latitude=21.0
    # )
    # print(f"找到 {len(ships)} 艘船舶")
    
    pass
    