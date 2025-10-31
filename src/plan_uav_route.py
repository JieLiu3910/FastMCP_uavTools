import os
import math
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_manager import load_config

uav_config = load_config()["uav_params"]


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度坐标之间的距离（先转换为东北天坐标系再计算欧几里得距离）
    """
    # 将第二个点的坐标作为参考点，转换为ENU坐标系
    east, north, up = lla_to_enu(lat1, lon1, 0, lat2, lon2, 0)

    # 在ENU坐标系下计算欧几里得距离
    distance = math.sqrt(east**2 + north**2 + up**2)
    return distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从点1到点2的方位角
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad

    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(dlon)

    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_destination(lat1, lon1, bearing, distance):
    """
    根据起始点、方位角和距离计算终点坐标
    """
    R = 6371000  # 地球半径（米）

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    bearing_rad = math.radians(bearing)

    distance_ratio = distance / R

    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(distance_ratio)
        + math.cos(lat1_rad) * math.sin(distance_ratio) * math.cos(bearing_rad)
    )

    lon2_rad = lon1_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_ratio) * math.cos(lat1_rad),
        math.cos(distance_ratio) - math.sin(lat1_rad) * math.sin(lat2_rad),
    )

    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)

    return lat2, lon2


def lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    将经纬高坐标转换为以参考点为原点的东北天(ENU)坐标系下的坐标

    参数:
    - lat, lon, alt: 目标点的经纬高
    - ref_lat, ref_lon, ref_alt: 参考点的经纬高

    返回:
    - east, north, up: ENU坐标系下的坐标（米）
    """
    # WGS-84椭球参数
    a = 6378137.0  # 长半轴
    b = 6356752.3142  # 短半轴
    f = (a - b) / a  # 扁率

    # 计算参考点的曲率半径
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    # 卯酉圈曲率半径
    N = a / math.sqrt(1 - f * (2 - f) * math.sin(ref_lat_rad) ** 2)

    # 计算目标点与参考点的差值
    delta_lat = math.radians(lat - ref_lat)
    delta_lon = math.radians(lon - ref_lon)
    delta_alt = alt - ref_alt

    # 转换为ENU坐标
    north = delta_lat * (N + ref_alt)
    east = delta_lon * (N + ref_alt) * math.cos(ref_lat_rad)
    up = delta_alt

    return east, north, up


def enu_to_lla(east, north, up, ref_lat, ref_lon, ref_alt):
    """
    将东北天(ENU)坐标转换为经纬高坐标

    参数:
    - east, north, up: ENU坐标系下的坐标（米）
    - ref_lat, ref_lon, ref_alt: 参考点的经纬高

    返回:
    - lat, lon, alt: 目标点的经纬高
    """
    # WGS-84椭球参数
    a = 6378137.0  # 长半轴
    b = 6356752.3142  # 短半轴
    f = (a - b) / a  # 扁率

    # 计算参考点的曲率半径
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    # 卯酉圈曲率半径
    N = a / math.sqrt(1 - f * (2 - f) * math.sin(ref_lat_rad) ** 2)

    # 转换为经纬高
    delta_lat = north / (N + ref_alt)
    delta_lon = east / ((N + ref_alt) * math.cos(ref_lat_rad))

    lat = ref_lat + math.degrees(delta_lat)
    lon = ref_lon + math.degrees(delta_lon)
    alt = ref_alt + up

    return lat, lon, alt


def calculate_uav_flight_time_to_target(
    uav_lat, uav_lon, uav_alt, target_lat, target_lon, target_alt, config
):
    """
    计算无人机飞往目标位置所需的时间（包含起飞模型）

    参数:
    - uav_lat, uav_lon, uav_alt: 无人机初始经纬高位置
    - target_lat, target_lon, target_alt: 目标位置经纬高
    - config: 配置参数字典

    返回:
    - 总飞行时间（秒）
    """
    # 从配置文件获取参数
    uav_max_speed = uav_config["uav_max_speed"]  # 无人机最大速度
    uav_cruise_speed = uav_config["uav_cruise_speed"]  # 无人机巡航速度
    uav_acceleration = uav_config["uav_acceleration"]  # 无人机加速度
    uav_deceleration = uav_config["uav_deceleration"]  # 无人机减速度
    uav_takeoff_climb_rate = uav_config["uav_takeoff_climb_rate"]  # 无人机起飞爬升率
    uav_cruise_alt = uav_config["uav_cruise_alt"]  # 无人机巡航高度
    uav_takeoff_runway_direction = uav_config[
        "uav_takeoff_runway_direction"
    ]  # 起飞跑道方向（度）

    # 计算水平距离
    horizontal_distance = calculate_distance(uav_lat, uav_lon, target_lat, target_lon)

    # 计算目标方位角
    target_bearing = calculate_bearing(uav_lat, uav_lon, target_lat, target_lon)

    # 计算高度差
    altitude_diff = uav_cruise_alt - uav_alt  # 无人机爬升到巡航高度

    # 如果已经在巡航高度，则直接飞往目标
    if abs(uav_alt - uav_cruise_alt) < 1.0:  # 高度差小于1米认为在同一高度
        # 直接水平飞行到目标
        time_to_target = horizontal_distance / uav_cruise_speed
        return time_to_target

    # 飞行阶段分析：
    # 1. 起飞滑跑阶段：沿跑道方向加速并爬升至巡航高度
    # 2. 巡航阶段：以巡航速度和高度飞向目标上方

    # 阶段1：起飞滑跑和爬升到巡航高度
    # 计算爬升所需时间和水平距离
    climb_altitude = uav_cruise_alt - uav_alt
    time_to_climb = climb_altitude / uav_takeoff_climb_rate

    # 计算爬升阶段的水平飞行距离（假设同时加速到巡航速度）
    # 简化模型：假设在爬升过程中同时加速到巡航速度
    # 使用匀加速运动公式计算加速距离
    time_to_accelerate_to_cruise = uav_cruise_speed / uav_acceleration
    distance_during_acceleration = (
        0.5 * uav_acceleration * time_to_accelerate_to_cruise**2
    )

    # 爬升阶段的水平距离
    horizontal_distance_during_takeoff = max(
        distance_during_acceleration, uav_cruise_speed * time_to_climb
    )

    # 阶段2：巡航飞行到目标点上方
    # 计算剩余的水平距离
    remaining_horizontal_distance = (
        horizontal_distance - horizontal_distance_during_takeoff
    )

    # 如果剩余距离为负，说明在起飞阶段就已经超过了目标点
    if remaining_horizontal_distance < 0:
        # 在这种情况下，我们需要重新计算飞行剖面
        # 简化处理：假设飞行路径是直线，直接计算飞行时间
        direct_distance = math.sqrt(horizontal_distance**2 + altitude_diff**2)
        time_to_target = direct_distance / uav_cruise_speed
        return time_to_target

    # 巡航阶段时间
    time_cruise = remaining_horizontal_distance / uav_cruise_speed

    total_time = time_to_climb + time_cruise
    return total_time


def plan_uav_path(
    uav_lat, uav_lon, uav_alt, target_lat, target_lon, target_alt, uav_config
):
    """
    规划无人机从初始位置到目标位置上方的飞行路径

    参数:
    - uav_lat, uav_lon, uav_alt: 无人机初始经纬高位置
    - target_lat, target_lon, target_alt: 目标位置经纬高
    - config: 配置参数字典

    返回:
    - 路径点列表，每个点包含经纬高和ENU坐标
    """
    # 从配置文件获取参数
    uav_cruise_speed = uav_config["uav_cruise_speed"]
    uav_takeoff_climb_rate = uav_config["uav_takeoff_climb_rate"]
    uav_cruise_alt = uav_config["uav_cruise_alt"]
    uav_takeoff_runway_direction = uav_config["uav_takeoff_runway_direction"]

    # 初始化路径点列表
    path_points = []

    # Add start point
    east, north, up = lla_to_enu(uav_lat, uav_lon, uav_alt, uav_lat, uav_lon, uav_alt)
    path_points.append(
        {
            "latitude": uav_lat,
            "longitude": uav_lon,
            "altitude": uav_alt,
            "east": east,
            "north": north,
            "up": up,
            "description": "Start Point",
        }
    )

    # # If not at cruise altitude, climb to cruise altitude first
    # if abs(uav_alt - uav_cruise_alt) >= 1.0:
    #     # Calculate climb phase path points
    #     climb_altitude = uav_cruise_alt - uav_alt
    #     time_to_climb = abs(climb_altitude) / uav_takeoff_climb_rate

    #     # Simplified approach: add one intermediate point to represent climbing to cruise altitude
    #     # In actual application, more points can be added as needed
    #     intermediate_alt = uav_cruise_alt
    #     intermediate_lat, intermediate_lon = calculate_destination(
    #         uav_lat, uav_lon, uav_takeoff_runway_direction, 10.0
    #     )  # Move 10 meters along the runway direction

    #     east, north, up = lla_to_enu(
    #         intermediate_lat,
    #         intermediate_lon,
    #         intermediate_alt,
    #         uav_lat,
    #         uav_lon,
    #         uav_alt,
    #     )
    #     path_points.append(
    #         {
    #             "latitude": intermediate_lat,
    #             "longitude": intermediate_lon,
    #             "altitude": intermediate_alt,
    #             "east": east,
    #             "north": north,
    #             "up": up,
    #             "description": "Climb Point",
    #         }
    #     )

    # 添加巡航阶段的路径点
    # 计算目标相对于起始点的方位和距离
    target_bearing = calculate_bearing(uav_lat, uav_lon, target_lat, target_lon)
    horizontal_distance = calculate_distance(uav_lat, uav_lon, target_lat, target_lon)

    # 在起始点和目标点之间添加几个中间点
    num_intermediate_points = 5
    for i in range(1, num_intermediate_points + 1):
        fraction = i / (num_intermediate_points + 1)
        distance = horizontal_distance * fraction
        intermediate_lat, intermediate_lon = calculate_destination(
            uav_lat, uav_lon, target_bearing, distance
        )
        intermediate_alt = uav_alt + uav_cruise_alt/num_intermediate_points*i

        east, north, up = lla_to_enu(
            intermediate_lat,
            intermediate_lon,
            intermediate_alt,
            uav_lat,
            uav_lon,
            uav_alt,
        )
        path_points.append(
            {
                "latitude": intermediate_lat,
                "longitude": intermediate_lon,
                "altitude": intermediate_alt,
                "east": east,
                "north": north,
                "up": up,
                "description": f"Cruise Point {i}",
            }
        )

    # Add point above the target position (maintaining cruise altitude)
    east, north, up = lla_to_enu(
        target_lat, target_lon, uav_cruise_alt, uav_lat, uav_lon, uav_alt
    )
    path_points.append(
        {
            "latitude": target_lat,
            "longitude": target_lon,
            "altitude": uav_cruise_alt,
            "east": east,
            "north": north,
            "up": up,
            "description": "Target Above",
        }
    )

    return path_points


def generate_spiral_waypoints(
    center_lat,
    center_lon,
    center_alt,
    radius,
    altitude,
    num_spirals,
    points_per_circle,
    ref_lat,
    ref_lon,
    ref_alt,
):
    """
    生成螺旋形侦察路径点

    参数:
    - center_lat, center_lon, center_alt: 螺旋中心点的经纬高
    - radius: 螺旋半径（米）
    - altitude: 飞行高度（米）
    - num_spirals: 螺旋圈数
    - points_per_circle: 每圈点数
    - ref_lat, ref_lon, ref_alt: 参考点的经纬高（用于ENU坐标转换）

    返回:
    - 螺旋路径点列表
    """
    waypoints = []

    # 生成螺旋路径点
    total_points = int(num_spirals * points_per_circle)
    for i in range(total_points):
        # 计算当前点的角度（弧度）
        angle = 2 * math.pi * i / points_per_circle

        # 计算当前点的半径（逐渐减小到0）
        current_radius = radius * (i / total_points) + 500.0  # (1 - i / total_points)

        # 计算当前点的ENU坐标（相对于螺旋中心点）
        east = current_radius * math.cos(angle)
        north = current_radius * math.sin(angle)
        up = 0  # 保持恒定高度

        # 将ENU坐标转换为经纬高坐标
        lat, lon, _ = enu_to_lla(east, north, up, center_lat, center_lon, center_alt)

        # 计算相对于参考点的ENU坐标
        ref_east, ref_north, ref_up = lla_to_enu(
            lat, lon, altitude, ref_lat, ref_lon, ref_alt
        )

        waypoints.append(
            {
                "latitude": lat,
                "longitude": lon,
                "altitude": altitude,
                "east": ref_east,
                "north": ref_north,
                "up": ref_up,
                "description": f"search {i+1}",
            }
        )
        # print(f"lat: {lat},lon: {lon}")
    return waypoints


def transform_spiral_waypoints(
    spiral_waypoints, uav_lat, uav_lon, uav_alt, vehicle_lat, vehicle_lon, vehicle_alt
):
    """
    对生成的螺旋线进行平移和旋转，使其和无人机与车辆初始位置连线平滑连接

    参数:
    - spiral_waypoints: 螺旋线路径点列表
    - uav_lat, uav_lon, uav_alt: 无人机初始位置
    - vehicle_lat, vehicle_lon, vehicle_alt: 车辆初始位置

    返回:
    - 变换后的螺旋线路径点列表
    """
    if not spiral_waypoints:
        return []

    # 计算无人机到车辆的方位角
    bearing = calculate_bearing(uav_lat, uav_lon, vehicle_lat, vehicle_lon)

    # 计算螺旋线的中心点（第一个点作为参考）
    spiral_center_lat = spiral_waypoints[0]["latitude"]
    spiral_center_lon = spiral_waypoints[0]["longitude"]
    spiral_center_alt = spiral_waypoints[0]["altitude"]

    # 计算需要平移到的目标位置（车辆位置）
    target_lat = vehicle_lat
    target_lon = vehicle_lon
    target_alt = vehicle_alt

    # 计算旋转角度（使螺旋线的起始方向对准无人机-车辆方向）
    # 螺旋线起始方向默认是正东方向（角度为0度）
    rotation_angle = bearing  # 旋转角度等于无人机到车辆的方位角

    # 变换后的路径点
    transformed_waypoints = []

    # 参考点（用于ENU坐标转换）
    ref_lat, ref_lon, ref_alt = uav_lat, uav_lon, uav_alt

    for i, point in enumerate(spiral_waypoints):
        # 1. 平移：将螺旋线中心移动到目标位置
        # 首先转换为ENU坐标
        east, north, up = lla_to_enu(
            point["latitude"],
            point["longitude"],
            point["altitude"],
            spiral_center_lat,
            spiral_center_lon,
            spiral_center_alt,
        )

        # 将ENU坐标转换回经纬高，以目标位置为中心
        temp_lat, temp_lon, temp_alt = enu_to_lla(
            east, north, up, target_lat, target_lon, target_alt
        )

        # 2. 旋转：绕目标位置旋转
        # 计算点相对于目标位置的ENU坐标
        point_east, point_north, point_up = lla_to_enu(
            temp_lat, temp_lon, temp_alt, target_lat, target_lon, target_alt
        )

        # 应用旋转变换
        rotated_east = point_east * math.cos(
            math.radians(rotation_angle)
        ) + point_north * math.sin(math.radians(rotation_angle))
        rotated_north = -point_east * math.sin(
            math.radians(rotation_angle)
        ) + point_north * math.cos(math.radians(rotation_angle))
        rotated_up = point_up  # 高度保持不变

        # 转换回经纬高坐标
        final_lat, final_lon, final_alt = enu_to_lla(
            rotated_east, rotated_north, rotated_up, target_lat, target_lon, target_alt
        )

        # 计算相对于参考点（无人机起始位置）的ENU坐标
        ref_east, ref_north, ref_up = lla_to_enu(
            final_lat, final_lon, final_alt, ref_lat, ref_lon, ref_alt
        )

        transformed_waypoints.append(
            {
                "latitude": final_lat,
                "longitude": final_lon,
                "altitude": spiral_center_alt,
                "east": ref_east,
                "north": ref_north,
                "up": spiral_center_alt,
                "description": f"search {i+1}",
            }
        )

    return transformed_waypoints


def generate_scan_waypoints(
    center_lat,
    center_lon,
    center_alt,
    search_radius,
    uav_turn_radius,
    altitude,
    fov_width,
    fov_height,
    ref_lat,
    ref_lon,
    ref_alt,
):
    """
    生成扫描式侦察路径点

    参数:
    - center_lat, center_lon, center_alt: 搜索区域中心点的经纬高
    - search_radius: 搜索区域半径（米）
    - uav_turn_radius: 无人机转弯半径（米）
    - altitude: 侦察飞行高度（米）
    - fov_width: 航拍视场宽度（米）
    - fov_height: 航拍视场高度（米）
    - ref_lat, ref_lon, ref_alt: 参考点的经纬高（用于ENU坐标转换）
    - uav_init_lat, uav_init_lon, uav_init_alt: 无人机初始位置（可选）

    返回:
    - 扫描路径点列表
    """
    waypoints = []

    # 计算扫描行数和列数
    # 考虑到转弯半径，行间距应略大于视场高度
    row_spacing = fov_height  # 适当重叠以确保覆盖
    col_spacing = search_radius * 2 + uav_turn_radius  # fov_width * 0.8

    # 计算需要扫描的行数和列数
    rows = int(math.ceil((search_radius * 2) / row_spacing))
    cols = 2  # int((search_radius * 2) / col_spacing) + 2

    # 计算搜索区域的四个顶点坐标 (ENU)
    vertices_enu = [
        (-search_radius, search_radius, 0),  # 左上角 (0)
        (search_radius, search_radius, 0),  # 右上角 (1)
        (search_radius, -search_radius, 0),  # 右下角 (2)
        (-search_radius, -search_radius, 0),  # 左下角 (3)
    ]

    # 将顶点ENU坐标转换为经纬高坐标
    vertices_lla = [
        enu_to_lla(e, n, u, center_lat, center_lon, center_alt)
        for e, n, u in vertices_enu
    ]

    # 默认参数
    start_north = search_radius
    start_east = -search_radius
    east_direction = 1  # 1表示从左到右，-1表示从右到左
    north_direction = -1  # -1表示从上到下，1表示从下到上

    # 如果提供了无人机初始位置，则计算到四个顶点的距离并选择最近的作为起始点
    # 计算无人机初始位置到四个顶点的距离
    distances = [
        calculate_distance(ref_lat, ref_lon, vert_lat, vert_lon)
        for vert_lat, vert_lon, _ in vertices_lla
    ]

    # 找到距离最小的顶点索引
    min_distance_index = distances.index(min(distances))
    start_vertex_enu = vertices_enu[min_distance_index]

    # 根据最近的顶点确定起始扫描方向
    start_north = start_vertex_enu[1]  # 北向坐标
    start_east = start_vertex_enu[0]  # 东向坐标
    # print(f"min_distance_index = {min_distance_index}, start_east = {start_east}, start_north = {start_north}")

    if min_distance_index == 0:
        east_direction = 1
        north_direction = -1
    elif min_distance_index == 1:
        east_direction = -1
        north_direction = -1
    elif min_distance_index == 2:
        east_direction = -1
        north_direction = 1
    else:
        east_direction = 1
        north_direction = 1

    # 生成扫描路径点
    for row in range(rows):
        # 计算当前行的北向坐标
        north = (
            start_north
            + row * row_spacing * north_direction
            + 0.5 * fov_height * north_direction
        )
        # print(f"Row {row}: north = {north}")  # 调试打印

        # # 根据行号确定east的基准值
        # if row == 0:
        #     # 第一行以start_east为基准
        #     base_east = start_east
        #     col_spacing = search_radius * 2 + uav_turn_radius
        # else:
        #     # 其他行以偏移后的值为基准，偏移量为uav_turn_radius*east_direction
        base_east = start_east - uav_turn_radius * east_direction
        col_spacing = search_radius * 2 + 2 * uav_turn_radius
        # print(f"base_east = {base_east}, row = {row}, east_direction = {east_direction}")
        # 交替扫描方向（根据起始顶点确定的扫描方向）
        if east_direction == 1:
            if row % 2 == 0:
                # 从左到右
                east_positions = [base_east + col * col_spacing for col in range(cols)]
                # print(f"Left_to_Right --- Row {row}: east_positions = {east_positions}")  # 调试打印
            else:
                # 从右到左
                east_positions = [
                    base_east + col * col_spacing for col in range(cols - 1, -1, -1)
                ]
                # print(f"Right_to_Left --- Row {row}: east_positions = {east_positions}")
        else:
            if row % 2 == 0:
                # 从右到左
                east_positions = [base_east - col * col_spacing for col in range(cols)]
                # print(f"Right_to_Left --- Row {row}: east_positions = {east_positions}")
            else:
                # 从左到右
                east_positions = [
                    base_east - col * col_spacing for col in range(cols - 1, -1, -1)
                ]
                # print(f"Left_to_Right --- Row {row}: east_positions = {east_positions}")  # 调试打印

        # 为当前行生成路径点
        for i, east in enumerate(east_positions):
            # print(f"Row {row}, Col {i}: east = {east}, north = {north}")  # 调试打印
            # 检查点是否在搜索区域内
            distance_from_center = math.sqrt(east**2 + north**2)
            if distance_from_center > 0:  # 修复：检查点是否在搜索半径内
                # 将ENU坐标转换为经纬高坐标
                lat_lon_alt = enu_to_lla(
                    east, north, 0, center_lat, center_lon, center_alt
                )
                lat, lon, _ = lat_lon_alt

                # 计算相对于参考点的ENU坐标
                rel_east, rel_north, rel_up = lla_to_enu(
                    lat, lon, altitude, ref_lat, ref_lon, ref_alt
                )
                # print(f"Row {row}, Col {i}: rel_east = {rel_east}, rel_north = {rel_north}")  # 调试打印

                waypoints.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "altitude": altitude,
                        "east": rel_east,
                        "north": rel_north,
                        "up": rel_up,
                        "description": f"search ({row+1},{i+1})",
                    }
                )

    return waypoints


def plot_uav_path(
    path_points, waypoints=None, shooting_points=None, search_area_params=None
):
    """
    绘制无人机路径

    参数:
    - path_points: 飞行路径点列表
    - waypoints: 侦察路径点列表（可选，可以是螺旋或扫描路径）
    - shooting_points: 航拍点列表（可选）
    - search_area_params: 搜索区域参数（可选，包含center_east, center_north, radius）
    """
    # 提取飞行路径的ENU坐标
    east_coords = [point["east"] for point in path_points]
    north_coords = [point["north"] for point in path_points]
    up_coords = [point["up"] for point in path_points]

    # 创建2D路径图
    plt.figure(figsize=(12, 5))

    # 子图1：水平面路径（东-北坐标）
    plt.subplot(1, 2, 1)
    plt.plot(east_coords, north_coords, "b-o", markersize=4, label="Flight Path")

    # 如果提供了搜索区域参数，绘制圆形区域和最小外接矩形
    if search_area_params:
        center_east = search_area_params["center_east"]
        center_north = search_area_params["center_north"]
        radius = search_area_params["radius"]

        # 绘制圆形区域
        circle = plt.Circle(
            (center_east, center_north),
            radius,
            color="gray",
            fill=False,
            linestyle="--",
            label="Search Area",
        )
        plt.gca().add_patch(circle)

        # 绘制最小外接矩形
        rect = plt.Rectangle(
            (center_east - radius, center_north - radius),
            2 * radius,
            2 * radius,
            fill=False,
            linestyle="-.",
            color="purple",
            label="Minimum Bounding Rectangle",
        )
        plt.gca().add_patch(rect)

        # 标注中心点
        plt.plot(center_east, center_north, "go", markersize=6, label="Search Center")

        # 标注矩形区域的顶点编号
        vertices = [
            (center_east - radius, center_north + radius),  # 左上角 (0)
            (center_east + radius, center_north + radius),  # 右上角 (1)
            (center_east + radius, center_north - radius),  # 右下角 (2)
            (center_east - radius, center_north - radius),  # 左下角 (3)
        ]

        for i, (x, y) in enumerate(vertices):
            plt.annotate(
                f"Vertex {i}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="purple",
            )

    # # 如果有侦察路径，也绘制
    # if waypoints:
    #     waypoint_east = [point['east'] for point in waypoints]
    #     waypoint_north = [point['north'] for point in waypoints]
    #     plt.plot(waypoint_east, waypoint_north, 'r-', linewidth=1, label='Reconnaissance Path')
    #     plt.scatter(waypoint_east[0], waypoint_north[0], color='green', s=50, marker='s', label='Reconnaissance Start')
    #     plt.scatter(waypoint_east[-1], waypoint_north[-1], color='red', s=50, marker='s', label='Reconnaissance End')

    # 如果有航拍点，也绘制
    if shooting_points:
        shooting_east = [point["east"] for point in shooting_points]
        shooting_north = [point["north"] for point in shooting_points]
        plt.scatter(
            shooting_east,
            shooting_north,
            color="orange",
            s=30,
            marker="x",
            label="Shooting Points",
        )

        # 标注前几个航拍点
        for i, (e, n) in enumerate(zip(shooting_east[:5], shooting_north[:5])):
            plt.annotate(
                f"Shooting Point {i+1}",
                (e, n),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                color="orange",
            )

    plt.xlabel("East Distance (m)")
    plt.ylabel("North Distance (m)")
    plt.title("UAV Horizontal Path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # 使坐标轴比例一致

    # 标注关键点
    for i, point in enumerate(path_points):
        if point["description"] in ["起始点", "目标点上方"]:
            label = (
                "Start Point" if point["description"] == "起始点" else "Target Above"
            )
            plt.annotate(
                label,
                (east_coords[i], north_coords[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # 子图2：高度剖面图
    distances = []
    total_distance = 0
    distances.append(total_distance)

    for i in range(1, len(path_points)):
        # 计算与前一个点的水平距离
        dx = east_coords[i] - east_coords[i - 1]
        dy = north_coords[i] - north_coords[i - 1]
        segment_distance = math.sqrt(dx**2 + dy**2)
        total_distance += segment_distance
        distances.append(total_distance)

    plt.subplot(1, 2, 2)
    plt.plot(distances, up_coords, "b-o", markersize=4, label="Flight Altitude")

    # 如果有搜索区域信息，在高度图上标注
    if search_area_params:
        # 简化处理：在高度图上标注搜索区域中心点的投影位置
        pass

    # 如果有航拍点，也在高度图上标注
    if shooting_points:
        # 计算航拍点在高度图上的位置（简化处理，假设所有航拍点都在同一高度）
        shooting_distances = []
        for point in shooting_points:
            # 简化处理：将航拍点放在飞行路径的末端
            shooting_distances.append(distances[-1])

        plt.scatter(
            shooting_distances,
            [point["up"] for point in shooting_points],
            color="orange",
            s=30,
            marker="x",
            label="Shooting Points",
        )

    plt.xlabel("Cumulative Horizontal Distance (m)")
    plt.ylabel("Altitude (m)")
    plt.title("UAV Altitude Profile")
    plt.legend()
    plt.grid(True)

    # 标注关键点
    for i, point in enumerate(path_points):
        if point["description"] in ["起始点", "目标点上方"]:
            label = (
                "Start Point" if point["description"] == "起始点" else "Target Above"
            )
            plt.annotate(
                label,
                (distances[i], up_coords[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig("uav_path.png", dpi=300, bbox_inches="tight")
    plt.close()


def uav_tracking_shooting(
    uav_lat: float,
    uav_lon: float,
    uav_alt: float,
    vehicle_lat: float,
    vehicle_lon: float,
    vehicle_alt: float,
    current_time: str,
    scan_mode: int = 0,
    out_dir: str = None,
):
    """
    计算无人机抵达车辆位置附近后开始航拍后各个航拍点的经纬高位置和对应时刻

    参数:
    - uav_lat, uav_lon, uav_alt: 无人机初始经纬高位置
    - vehicle_lat, vehicle_lon, vehicle_alt: 车辆初始经纬高位置
    - current_time: 当前时刻（datetime.time对象）
    - scan_mode: 是否使用扫描式侦察路径（默认为False，使用螺旋式）

    返回:
    - 航拍点列表，每个元素包含(经纬高位置, 时间)
    """

    # 默认路径
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results/uav_way"
    )
    out_file = os.path.join(out_dir, "uav_way.json")

    # 读取配置文件

    # config = load_config()["uav_params"]

    # 获取配置参数
    uav_cruise_alt = uav_config["uav_cruise_alt"]  # 无人机巡航高度
    uav_max_speed = uav_config.get("uav_max_speed", 10.0)
    uav_cruise_speed = uav_config.get("uav_cruise_speed", 10.0)
    uav_turn_radius = uav_config.get("uav_turn_radius", 100.0)  # 无人机转弯半径
    fov_width = uav_config.get("shooting_fov_width", 100.0)  # 视场长度
    fov_height = uav_config.get("shooting_fov_height", 75.0)  # 视场宽度
    shooting_maxnum = uav_config.get("shooting_maxnum", 5)
    target_max_speed = uav_config.get("target_max_speed", 10.0)  # 目标最大速度
    time_interval = uav_config.get("time_interval", 0.5)  # 插值平滑时间间隔
    scan_mode = uav_config.get("scan_mode", 1)

    print("time_interval: ", time_interval)
    # 规划无人机路径到目标点上方
    uav_path = plan_uav_path(
        uav_lat,
        uav_lon,
        uav_alt,
        vehicle_lat,
        vehicle_lon,
        uav_cruise_alt,  # 飞到目标点上方，保持巡航高度
        uav_config,
    )

    # # 绘制路径
    # plot_uav_path(uav_path)

    # 计算无人机到达车辆初始位置所需时间（使用改进的起飞模型）
    flight_time_to_target = calculate_uav_flight_time_to_target(
        uav_lat,
        uav_lon,
        uav_alt,
        vehicle_lat,
        vehicle_lon,
        uav_cruise_alt,  # 飞到目标点上方，保持巡航高度
        uav_config,
    )
    distance_uav2vehicle = calculate_distance(
        uav_lat, uav_lon, vehicle_lat, vehicle_lon
    )
    # 计算无人机抵达车辆位置上方时的时间
    # current_time = datetime.strptime(current_time, "%H:%M:%S").time()
    start_datetime = datetime.combine(datetime.today(), current_time)
    arrival_time = start_datetime + timedelta(seconds=flight_time_to_target)

    # 获取其他配置参数
    shooting_duration = uav_config["shooting_duration"]  # 单点航拍时间
    # 使用目标最大速度计算目标移动范围
    # 检查根号内的值是否为负数
    sqrt_expr = (
        fov_height * uav_cruise_speed
    ) ** 2 - 16 * fov_height * target_max_speed**2 * distance_uav2vehicle
    if sqrt_expr >= 0:
        # 如果表达式非负，则使用原始公式
        vehicle_distance_during_shooting = (
            fov_height * uav_cruise_speed + math.sqrt(sqrt_expr)
        ) / (8 * target_max_speed)
    else:
        # 如果表达式为负数，则使用简化计算方法
        # 这种情况下使用基于时间的简单距离计算
        vehicle_distance_during_shooting = target_max_speed * flight_time_to_target * 2
        # print(f"警告: 复杂距离计算公式产生负数根，使用简化计算方法。根表达式值: {sqrt_expr}")
    # 使用扫描式侦察路径
    # 计算目标移动的圆形区域范围半径
    search_radius = vehicle_distance_during_shooting
    if scan_mode == 0:

        # 生成扫描侦察路径点
        scan_waypoints = generate_scan_waypoints(
            vehicle_lat,
            vehicle_lon,
            uav_cruise_alt,
            search_radius,
            uav_turn_radius,
            uav_cruise_alt,
            fov_width,
            fov_height,
            uav_lat,
            uav_lon,
            uav_alt,  # 参考点为无人机起始位置
        )

        # 计算搜索区域中心点的ENU坐标
        center_east, center_north, _ = lla_to_enu(
            vehicle_lat, vehicle_lon, uav_cruise_alt, uav_lat, uav_lon, uav_alt
        )
        rect_lat1, rect_lon1, _ = enu_to_lla(
            center_east - search_radius,
            center_north + search_radius,
            uav_cruise_alt,
            uav_lat,
            uav_lon,
            uav_alt,
        )
        rect_lat2, rect_lon2, _ = enu_to_lla(
            center_east + search_radius,
            center_north + search_radius,
            uav_cruise_alt,
            uav_lat,
            uav_lon,
            uav_alt,
        )
        rect_lat3, rect_lon3, _ = enu_to_lla(
            center_east + search_radius,
            center_north - search_radius,
            uav_cruise_alt,
            uav_lat,
            uav_lon,
            uav_alt,
        )
        rect_lat4, rect_lon4, _ = enu_to_lla(
            center_east - search_radius,
            center_north - search_radius,
            uav_cruise_alt,
            uav_lat,
            uav_lon,
            uav_alt,
        )
        # 搜索区域参数
        search_area_params = {
            "center_east": center_east,
            "center_north": center_north,
            "radius": search_radius,
            "center_lat": vehicle_lat,
            "center_lon": vehicle_lon,
            "rect_lat1": rect_lat1,
            "rect_lon1": rect_lon1,
            "rect_lat2": rect_lat2,
            "rect_lon2": rect_lon2,
            "rect_lat3": rect_lat3,
            "rect_lon3": rect_lon3,
            "rect_lat4": rect_lat4,
            "rect_lon4": rect_lon4,
        }
        # save_search_area_params_to_json(search_area_params, "search_area_params.json")

        uav_path = []
        # Add start point
        east, north, up = lla_to_enu(
            uav_lat, uav_lon, uav_alt, uav_lat, uav_lon, uav_alt
        )
        uav_path.append(
            {
                "latitude": uav_lat,
                "longitude": uav_lon,
                "altitude": uav_alt,
                "east": east,
                "north": north,
                "up": up,
                "description": "Start Point",
            }
        )
        uav_path.extend(scan_waypoints)

        # 对扫描路径进行平滑处理
        smoothed_scan_waypoints = smooth_uav_path_with_bezier(
            uav_path, uav_cruise_speed, time_interval
        )
        searchstart = []
        Idx0 = 0
        for index, waypoint in enumerate(smoothed_scan_waypoints):
            if "search" in waypoint.get("description", ""):
                searchstart.append(waypoint)
                Idx0 = index
                print(
                    f"Found 'search' point at index {index} in smoothed_scan_waypoints"
                )
                break

        # 按配置文件中的航拍时间间隔参数，等时间间隔记录航拍点
        shooting_points = []
        time_interval = shooting_duration  # 航拍时间间隔

        # 在平滑后的扫描路径上等时间间隔采样
        num_shooting_points = min(
            len(smoothed_scan_waypoints) - Idx0,
            int((len(smoothed_scan_waypoints) - Idx0) / time_interval),
        )
        if num_shooting_points > shooting_maxnum:
            time_interval = time_interval * num_shooting_points / shooting_maxnum
            num_shooting_points = shooting_maxnum

        dn = int((len(smoothed_scan_waypoints) - Idx0) / (num_shooting_points + 1))

        for i in range(0, num_shooting_points):
            index = int((i + 1) * dn) + Idx0
            if index >= len(smoothed_scan_waypoints):
                index = len(smoothed_scan_waypoints) - 1
            print(f"index: {index}")
            point = smoothed_scan_waypoints[index]

            # 检查点是否在侦察区域内
            distance_from_center = math.sqrt(
                (point["east"] - center_east) ** 2
                + (point["north"] - center_north) ** 2
            )
            # if distance_from_center <= 0.9*search_radius:
            # 计算航拍点时间
            # point_time = arrival_time + timedelta(seconds=i * time_interval)
            # point_time = flight_time_to_target + i * time_interval + time_adjustment
            shooting_point = {
                "latitude": point["latitude"],
                "longitude": point["longitude"],
                "altitude": point["altitude"],
                "east": point["east"],
                "north": point["north"],
                "up": point["up"],
                "time": point["time"],
            }

            shooting_points.append(shooting_point)
            # print(f"---------------shooting_point: {shooting_point['time']}")
            # else:
            #     # 若点落在侦察区域外，则沿平滑路径往前查找，找到一个落在区域内的点
            #     # 更新航拍点位置和时间
            #     found_point = None
            #     found_index = index

            #     # 确定查找范围：前后两个航拍点之间
            #     start_index = max(Idx0, index - dn)
            #     end_index = min(len(smoothed_scan_waypoints), index + dn)

            #     # print(f"index: {index},search_radius: {search_radius},start_index = {start_index},end_index = {end_index}")

            #     # 向前查找（索引增加方向）
            #     for j in range(start_index + 1, end_index):
            #         next_point = smoothed_scan_waypoints[j]
            #         distance = math.sqrt((next_point['east'] - center_east)**2 + (next_point['north'] - center_north)**2)
            #         if distance <= 0.9*search_radius:
            #             found_point = next_point
            #             found_index = j
            #             break

            #     # # 如果前面没有找到，则向后查找（索引减小方向）
            #     # if found_point is None:
            #     #     for j in range(index - 1, start_index - 1, -1):
            #     #         prev_point = smoothed_scan_waypoints[j]
            #     #         distance = math.sqrt((prev_point['east'] - center_east)**2 + (prev_point['north'] - center_north)**2)
            #     #         if distance <= search_radius:
            #     #             found_point = prev_point
            #     #             found_index = j
            #     #             break

            #     # 检查找到的点是否距离相邻航拍点太近
            #     is_too_close = False
            #     if found_point is not None:
            #         # 检查与前一个航拍点的距离
            #         if found_index > Idx0:
            #             prev_shooting_point = smoothed_scan_waypoints[index - dn]
            #             distance_to_prev = math.sqrt(
            #                 (found_point['east'] - prev_shooting_point['east'])**2 +
            #                 (found_point['north'] - prev_shooting_point['north'])**2
            #             )
            #             if distance_to_prev < 0.1 * search_radius:  # 距离小于搜索半径的10%认为太近
            #                 is_too_close = True

            #         # 检查与后一个航拍点的距离
            #         if found_index < len(smoothed_scan_waypoints) - 1:
            #             next_shooting_point = smoothed_scan_waypoints[index + dn]
            #             distance_to_next = math.sqrt(
            #                 (found_point['east'] - next_shooting_point['east'])**2 +
            #                 (found_point['north'] - next_shooting_point['north'])**2
            #             )
            #             if distance_to_next < 0.1 * search_radius:  # 距离小于搜索半径的10%认为太近
            #                 is_too_close = True

            #     # 如果找到在区域内的点且不与相邻航拍点太近，则更新航拍点
            #     if found_point is not None and not is_too_close:
            #         # 根据找到点的新索引调整时间
            #         time_adjustment = ((found_index - index) * time_interval /
            #                           (len(smoothed_scan_waypoints) - 1) * num_shooting_points)
            #         # point_time = arrival_time + timedelta(seconds=(i * time_interval + time_adjustment))
            #         point_time = flight_time_to_target + i * time_interval + time_adjustment
            #         shooting_point = {
            #             'latitude': found_point['latitude'],
            #             'longitude': found_point['longitude'],
            #             'altitude': found_point['altitude'],
            #             'east': found_point['east'],
            #             'north': found_point['north'],
            #             'up': found_point['up'],
            #             'time': found_point['time']
            #         }

            #         shooting_points.append(shooting_point)
            #         print(f"shooting_point: {shooting_point['time']}")

        data_result = save_smoothed_waypoints_to_json(
            smoothed_scan_waypoints, shooting_points, out_file
        )

        # # 更新路径图，包含航拍点
        # plot_uav_path(uav_path, scan_waypoints, shooting_points, search_area_params)
        # plot_smoothed_uav_path(uav_path, smoothed_scan_waypoints,shooting_points, search_area_params)
        # 添加起飞时间和到达时间信息
        result = {
            "flight_time": flight_time_to_target,
            "arrival_time": arrival_time.strftime("%H:%M:%S"),
            "waypoints_count": len(smoothed_scan_waypoints),
            "waypoints": uav_path,
            "scan_waypoints": smoothed_scan_waypoints,
        }

        return data_result
    else:
        # 使用螺旋式侦察路径（原有逻辑）
        # 生成螺旋侦察路径点
        # 螺旋参数
        spiral_radius = (
            vehicle_distance_during_shooting  # 螺旋半径为车辆在航拍时间内移动距离的一半
        )
        num_spirals = (int)(
            1.2 * vehicle_distance_during_shooting / fov_height
        )  # 3  # 螺旋圈数
        points_per_circle = 20  # 每圈点数

        # 生成螺旋侦察路径
        spiral_waypoints = generate_spiral_waypoints(
            vehicle_lat,
            vehicle_lon,
            uav_cruise_alt,
            spiral_radius,
            uav_cruise_alt,
            num_spirals,
            points_per_circle,
            uav_lat,
            uav_lon,
            uav_alt,  # 参考点为无人机起始位置
        )
        # 对螺旋线进行平移和旋转，使其和无人机与车辆初始位置连线平滑连接
        transformed_spiral_waypoints = transform_spiral_waypoints(
            spiral_waypoints,
            uav_lat,
            uav_lon,
            uav_alt,
            vehicle_lat,
            vehicle_lon,
            vehicle_alt,
        )

        # 如果变换成功，则使用变换后的路径点
        if transformed_spiral_waypoints:
            spiral_waypoints = transformed_spiral_waypoints

        # # 对螺旋路径进行平滑处理
        # smoothed_spiral_waypoints = smooth_uav_path_with_bezier(spiral_waypoints, uav_cruise_speed, 1.0)

        # 计算搜索区域中心点的ENU坐标
        center_east, center_north, _ = lla_to_enu(
            vehicle_lat, vehicle_lon, uav_cruise_alt, uav_lat, uav_lon, uav_alt
        )

        # 搜索区域参数
        search_area_params = {
            "center_east": center_east,
            "center_north": center_north,
            "radius": search_radius,
            "center_lat": vehicle_lat,
            "center_lon": vehicle_lon,
        }
        # # # 更新路径图，包含螺旋侦察路径和搜索区域
        # plot_uav_path(uav_path, spiral_waypoints, search_area_params=search_area_params)

        uav_path.extend(spiral_waypoints)
        smoothed_scan_waypoints = smooth_uav_path_with_lineinter(
            uav_path, uav_cruise_speed, time_interval
        )
        searchstart = []
        Idx0 = 0
        for index, waypoint in enumerate(smoothed_scan_waypoints):
            if "search" in waypoint.get("description", ""):
                searchstart.append(waypoint)
                Idx0 = index
                print(
                    f"Found 'search' point at index {index} in smoothed_scan_waypoints"
                )
                break
        # 按配置文件中的航拍时间间隔参数，等时间间隔记录航拍点
        shooting_points = []
        time_interval = shooting_duration  # 航拍时间间隔

        # 计算螺旋路径总时间（假设匀速飞行）
        spiral_circumference = 2 * math.pi * spiral_radius
        total_spiral_distance = num_spirals * spiral_circumference
        uav_cruise_speed = uav_config["uav_cruise_speed"]
        total_spiral_time = total_spiral_distance / uav_cruise_speed

        # 在平滑后的螺旋路径上等时间间隔采样
        num_shooting_points = int(total_spiral_time / time_interval)
        if num_shooting_points > shooting_maxnum:
            time_interval = time_interval * num_shooting_points / shooting_maxnum
            num_shooting_points = shooting_maxnum

        # num_shooting_points = int(total_spiral_time / time_interval)
        # if num_shooting_points > len(spiral_waypoints):
        #     num_shooting_points = len(spiral_waypoints)
        dn = int((len(smoothed_scan_waypoints) - Idx0) / (num_shooting_points + 1))
        for i in range(0, num_shooting_points):
            index = int((i + 1) * dn) + Idx0
            if index >= len(smoothed_scan_waypoints):
                index = len(smoothed_scan_waypoints) - 1

            point = smoothed_scan_waypoints[index]

            # # 检查点是否在侦察区域内（螺旋形搜索区域为圆形）
            # distance_from_center = math.sqrt((point['east'] - center_east)**2 + (point['north'] - center_north)**2)
            # if distance_from_center <= spiral_radius:
            #     # 计算航拍点时间
            #     # point_time = arrival_time + timedelta(seconds=i * time_interval)
            # point_time = flight_time_to_target + i * time_interval
            shooting_point = {
                "latitude": point["latitude"],
                "longitude": point["longitude"],
                "altitude": point["altitude"],
                "east": point["east"],
                "north": point["north"],
                "up": point["up"],
                "time": point["time"],
            }

            shooting_points.append(shooting_point)

        save_smoothed_waypoints_to_json(
            smoothed_scan_waypoints, shooting_points, out_file
        )

        # # 更新路径图，包含航拍点和搜索区域
        # plot_uav_path(uav_path, spiral_waypoints, shooting_points, search_area_params)
        # 添加起飞时间和到达时间信息
        result = {
            "flight_time": flight_time_to_target,
            "arrival_time": arrival_time.strftime("%H:%M:%S"),
            "waypoints_count": len(smoothed_scan_waypoints),
            "waypoints": uav_path,
            "scan_waypoints": spiral_waypoints,
        }
        return result


def interpolate_path_equidistant(path_points, min_distance, additional_distance):
    """
    对路径进行插值，在相邻路径点之间根据距离添加额外的点

    参数:
    - path_points: 原始路径点列表
    - min_distance: 最小距离阈值，当相邻点距离大于此值时添加额外点
    - additional_distance: 额外点距离端点的距离

    返回:
    - 插值后的路径点列表
    """
    if len(path_points) < 2:
        return path_points

    # 新的插值方法：根据距离判断是否添加额外点
    interpolated_points = []

    # 添加第一个点
    interpolated_points.append(path_points[0])
    interpolated_points[0]["description"] = path_points[0]["description"]

    # print(f"  Point {0}: description={interpolated_points[0]['description']}")
    # 在每对相邻点之间根据距离添加点
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]

        # 计算两点之间的距离
        segment_distance = calculate_distance(
            p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
        )

        if i == 0:
            for j in range(1, 4):
                fraction = j / 4
                # 线性插值计算中间点
                lat_fra = p1["latitude"] + fraction * (p2["latitude"] - p1["latitude"])
                lon_fra = p1["longitude"] + fraction * (
                    p2["longitude"] - p1["longitude"]
                )
                alt_fra = p1["altitude"] + fraction * (p2["altitude"] - p1["altitude"])
                interpolated_points.append(
                    {
                        "latitude": lat_fra,
                        "longitude": lon_fra,
                        "altitude": alt_fra,
                        "description": path_points[i]["description"],
                    }
                )

        elif segment_distance > min_distance:
            # elif abs(p1['latitude'] -p2['latitude'])< 0.00001:
            # 如果距离大于阈值，添加三个点：
            # 1. 距离p1点additional_distance的点
            # 2. 中点
            # 3. 距离p2点additional_distance的点

            # 计算方位角
            bearing = calculate_bearing(
                p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
            )

            # 添加距离p1点additional_distance的点
            near_lat, near_lon = calculate_destination(
                p1["latitude"], p1["longitude"], bearing, additional_distance
            )
            near_alt = p1["altitude"] + (additional_distance / segment_distance) * (
                p2["altitude"] - p1["altitude"]
            )

            interpolated_points.append(
                {
                    "latitude": near_lat,
                    "longitude": near_lon,
                    "altitude": near_alt,
                    "description": path_points[i]["description"],
                }
            )

            # 添加中点
            mid_lat, mid_lon = calculate_destination(
                p1["latitude"], p1["longitude"], bearing, segment_distance / 2
            )
            mid_alt = (p1["altitude"] + p2["altitude"]) / 2

            interpolated_points.append(
                {
                    "latitude": mid_lat,
                    "longitude": mid_lon,
                    "altitude": mid_alt,
                    "description": path_points[i]["description"],
                }
            )

            # 添加距离p2点additional_distance的点
            far_lat, far_lon = calculate_destination(
                p1["latitude"],
                p1["longitude"],
                bearing,
                segment_distance - additional_distance,
            )
            far_alt = p1["altitude"] + (
                (segment_distance - additional_distance) / segment_distance
            ) * (p2["altitude"] - p1["altitude"])

            interpolated_points.append(
                {
                    "latitude": far_lat,
                    "longitude": far_lon,
                    "altitude": far_alt,
                    "description": path_points[i]["description"],
                }
            )
        else:
            # 如果距离小于等于阈值，只添加中点
            mid_lat = (p1["latitude"] + p2["latitude"]) / 2
            mid_lon = (p1["longitude"] + p2["longitude"]) / 2
            mid_alt = (p1["altitude"] + p2["altitude"]) / 2

            interpolated_points.append(
                {
                    "latitude": mid_lat,
                    "longitude": mid_lon,
                    "altitude": mid_alt,
                    "description": path_points[i]["description"],
                }
            )

        # 添加第二个点
        interpolated_points.append(p2)
        interpolated_points[-1]["description"] = path_points[i + 1]["description"]

    # for i, point in enumerate(interpolated_points):
    #     print(f"  Point {i}: description={point['description']}, lat={point['latitude']}, lon={point['longitude']}, alt={point['altitude']}")

    #     print("interpolated_points:")
    # for i, point in enumerate(interpolated_points):
    #     print(f"  Point {i}: lat={point['latitude']}, lon={point['longitude']}, alt={point['altitude']}")
    # print(f"Total interpolated points: {len(interpolated_points)}")

    return interpolated_points


def smooth_uav_path_with_bezier(path_points, uav_speed, time_interval):
    """
    使用贝塞尔曲线对无人机路径进行平滑处理

    参数:
    - path_points: 原始路径点列表，每个点包含经纬高坐标
    - uav_speed: 无人机飞行速度（米/秒）
    - time_interval: 输出路径点的时间间隔（秒）

    返回:
    - 平滑后的路径点列表，按指定时间间隔输出
    """
    try:
        turn_radius = uav_config["turn_radius"]  # 默认100米
    except:
        turn_radius = 100.0  # 如果无法读取配置文件，则使用默认值
    # 处理嵌套列表的情况
    if len(path_points) > 0 and isinstance(path_points[0], list):
        # 展开嵌套列表
        flattened_path_points = []
        for item in path_points:
            if isinstance(item, list):
                flattened_path_points.extend(item)
            else:
                flattened_path_points.append(item)
        path_points = flattened_path_points

    if len(path_points) < 2:
        return path_points
    # print(f"Turn radius for bezier smoothing: {turn_radius}")

    # 先对原始路径进行插值
    interpolated_points = interpolate_path_equidistant(path_points, 5000.0, 2000.0)
    # print(f"Interpolated path points: {len(interpolated_points)}")

    # 平滑后的路径点列表
    smoothed_path = []

    # 添加第一个点
    first_point = interpolated_points[0].copy()
    first_point["curtime"] = 0.0
    smoothed_path.append(first_point)

    # 如果插值后的点少于4个，直接使用线性插值
    if len(interpolated_points) < 4:
        if len(interpolated_points) == 2 or len(interpolated_points) == 3:
            for i in range(len(interpolated_points) - 1):
                p1 = interpolated_points[i]
                p2 = interpolated_points[i + 1]

                # 计算两点之间的距离
                dist = calculate_distance(
                    p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
                )

                # 直线段上的点
                segment_time = dist / uav_speed
                num_points = max(1, int(segment_time / time_interval))
                curtime = 0.0
                # 线性插值计算中间点
                for j in range(1, num_points + 1):
                    fraction = j / (num_points + 1)
                    # 线性插值计算中间点
                    lat = p1["latitude"] + fraction * (p2["latitude"] - p1["latitude"])
                    lon = p1["longitude"] + fraction * (
                        p2["longitude"] - p1["longitude"]
                    )
                    alt = p1["altitude"] + fraction * (p2["altitude"] - p1["altitude"])

                    # 计算ENU坐标
                    ref_lat = smoothed_path[0]["latitude"]
                    ref_lon = smoothed_path[0]["longitude"]
                    ref_alt = smoothed_path[0]["altitude"]
                    east, north, up = lla_to_enu(
                        lat, lon, alt, ref_lat, ref_lon, ref_alt
                    )

                    # 计算smoothed_path中最后一个点和当前插值点之间的距离
                    distance_to_last_point = 0.0
                    if len(smoothed_path) > 0:
                        last_point = smoothed_path[-1]
                        distance_to_last_point = math.sqrt(
                            (east - last_point["east"]) ** 2
                            + (north - last_point["north"]) ** 2
                            + (up - last_point["up"]) ** 2
                        )

                    pathdict = pathdict + distance_to_last_point
                    curtime = pathdict / uav_speed

                    smoothed_path.append(
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "altitude": alt,
                            "east": east,
                            "north": north,
                            "up": up,
                            "time": curtime,
                            "pathdict": pathdict,
                            "description": interpolated_points[i]["description"],
                        }
                    )

        # 添加最后一个点
        if len(interpolated_points) > 1:
            last_point = interpolated_points[-1].copy()
            last_point["description"] = interpolated_points[-1]["description"]
            smoothed_path.append(last_point)

        return smoothed_path

    # 使用插值后的点构造三次贝塞尔曲线
    # 按每4个点为一组构造三次贝塞尔曲线
    i = 0
    curtime = 0.0
    pathdict = 0.0
    while i < len(interpolated_points) - 3:
        # 获取四个连续的点用于构造三次贝塞尔曲线
        p0 = interpolated_points[i]
        p1 = interpolated_points[i + 1]
        p2 = interpolated_points[i + 2]
        p3 = interpolated_points[i + 3]

        # 计算曲线段的总距离（近似）
        dist1 = calculate_distance(
            p0["latitude"], p0["longitude"], p1["latitude"], p1["longitude"]
        )
        dist2 = calculate_distance(
            p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
        )
        dist3 = calculate_distance(
            p2["latitude"], p2["longitude"], p3["latitude"], p3["longitude"]
        )
        total_dist = dist1 + dist2 + dist3

        # 计算曲线段的飞行时间
        curve_time = total_dist / uav_speed
        num_points = max(2, int(curve_time / time_interval))

        # 使用三次贝塞尔曲线生成点
        # B(t) = (1-t)³*P0 + 3*(1-t)²*t*P1 + 3*(1-t)*t²*P2 + t³*P3
        kk = i
        for j in range(num_points + 1):
            t = j / num_points

            # 三次贝塞尔曲线公式
            lat = (
                (1 - t) ** 3 * p0["latitude"]
                + 3 * (1 - t) ** 2 * t * p1["latitude"]
                + 3 * (1 - t) * t**2 * p2["latitude"]
                + t**3 * p3["latitude"]
            )

            lon = (
                (1 - t) ** 3 * p0["longitude"]
                + 3 * (1 - t) ** 2 * t * p1["longitude"]
                + 3 * (1 - t) * t**2 * p2["longitude"]
                + t**3 * p3["longitude"]
            )

            alt = (
                (1 - t) ** 3 * p0["altitude"]
                + 3 * (1 - t) ** 2 * t * p1["altitude"]
                + 3 * (1 - t) * t**2 * p2["altitude"]
                + t**3 * p3["altitude"]
            )

            # 计算ENU坐标
            ref_lat = smoothed_path[0]["latitude"]
            ref_lon = smoothed_path[0]["longitude"]
            ref_alt = smoothed_path[0]["altitude"]

            # east, north, up = lla_to_enu(interpolated_points[kk]['latitude'], interpolated_points[kk]['longitude'], interpolated_points[kk]['altitude'], ref_lat, ref_lon, ref_alt)
            # if len(smoothed_path) > 0:
            dict_interpoint_kk = calculate_distance(
                lat,
                lon,
                interpolated_points[kk]["latitude"],
                interpolated_points[kk]["longitude"],
            )
            # print(f"kk:{kk},dict_interpoint_kk:{dict_interpoint_kk}")
            if dict_interpoint_kk < 25.0:
                # print(f"kk:{kk},dict_interpoint_kk:{dict_interpoint_kk}")
                if kk <= i + 3:
                    kk = kk + 1
            # 计算ENU坐标

            east, north, up = lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

            # 计算smoothed_path中最后一个点和当前插值点之间的距离
            distance_to_last_point = 0.0
            if len(smoothed_path) > 0:
                last_point = smoothed_path[-1]
                distance_to_last_point = calculate_distance(
                    lat, lon, last_point["latitude"], last_point["longitude"]
                )

            pathdict = pathdict + distance_to_last_point
            curtime = pathdict / uav_speed

            if curtime > 0.1:
                smoothed_path.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "altitude": alt,
                        "east": east,
                        "north": north,
                        "up": up,
                        "time": curtime,
                        "pathdict": pathdict,
                        "description": interpolated_points[kk]["description"],
                    }
                )

        # 移动到下一组点（每次移动3个点，保证连接的连续性）
        i += 3

    # 处理剩余的点
    while i < len(interpolated_points):
        # 对于剩余的点，使用直线插值
        if i > 0 and i < len(interpolated_points):
            p1 = interpolated_points[i - 1]  # 前一个点
            p2 = interpolated_points[i]  # 当前点

            # 计算两点之间的距离
            dist = calculate_distance(
                p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
            )

            # 直线段上的点
            segment_time = dist / uav_speed
            num_points = max(1, int(segment_time / time_interval))

            # 线性插值计算中间点
            for j in range(1, num_points + 1):
                fraction = j / (num_points + 1)
                # 线性插值计算中间点
                lat = p1["latitude"] + fraction * (p2["latitude"] - p1["latitude"])
                lon = p1["longitude"] + fraction * (p2["longitude"] - p1["longitude"])
                alt = p1["altitude"] + fraction * (p2["altitude"] - p1["altitude"])

                # 计算ENU坐标
                ref_lat = smoothed_path[0]["latitude"]
                ref_lon = smoothed_path[0]["longitude"]
                ref_alt = smoothed_path[0]["altitude"]
                east, north, up = lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

                # 计算smoothed_path中最后一个点和当前插值点之间的距离
                distance_to_last_point = 0.0
                if len(smoothed_path) > 0:
                    last_point = smoothed_path[-1]
                    distance_to_last_point = math.sqrt(
                        (east - last_point["east"]) ** 2
                        + (north - last_point["north"]) ** 2
                        + (up - last_point["up"]) ** 2
                    )

                pathdict = pathdict + distance_to_last_point
                curtime = pathdict / uav_speed

                smoothed_path.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "altitude": alt,
                        "east": east,
                        "north": north,
                        "up": up,
                        "time": curtime,
                        "pathdict": pathdict,
                        "description": interpolated_points[i - 1]["description"],
                    }
                )
        i += 1

    # 添加最后一个点（如果还没有添加）
    if len(interpolated_points) > 1 and (
        len(smoothed_path) == 0
        or smoothed_path[-1]["latitude"] != interpolated_points[-1]["latitude"]
    ):
        last_point = interpolated_points[-1].copy()

        distance_to_last_point = 0.0
        if len(smoothed_path) > 0:
            distance_to_last_point = math.sqrt(
                (east - last_point["east"]) ** 2
                + (north - last_point["north"]) ** 2
                + (up - last_point["up"]) ** 2
            )

        pathdict = pathdict + distance_to_last_point
        curtime = pathdict / uav_speed

        last_point["pathdict"] = pathdict
        last_point["time"] = curtime
        last_point["description"] = interpolated_points[-1]["description"]
        smoothed_path.append(last_point)
    # return interpolated_points
    return smoothed_path


def smooth_uav_path_with_lineinter(path_points, uav_speed, time_interval, uav_config=None):
    """
    使用线性插值对无人机路径进行处理

    参数:
    - path_points: 原始路径点列表，每个点包含经纬高坐标
    - uav_speed: 无人机飞行速度（米/秒）
    - time_interval: 输出路径点的时间间隔（秒）

    返回:
    - 插值后的路径点列表，按指定时间间隔输出
    """
    # 处理嵌套列表的情况
    if len(path_points) > 0 and isinstance(path_points[0], list):
        # 展开嵌套列表
        flattened_path_points = []
        for item in path_points:
            if isinstance(item, list):
                flattened_path_points.extend(item)
            else:
                flattened_path_points.append(item)
        path_points = flattened_path_points

    if len(path_points) < 2:
        return path_points

    # 从配置文件读取参数
    try:
        turn_radius = uav_config["uav_turn_radius"]  # 默认100米
    except:
        turn_radius = 100.0  # 如果无法读取配置文件，则使用默认值

    # print(f"Turn radius for bezier smoothing: {turn_radius}")

    # 先对原始路径进行插值
    interpolated_points = interpolate_path_equidistant(path_points, 3000.0, 2000.0)
    # # 打印interpolated_points的内容
    # print("interpolated_points:")
    # for i, point in enumerate(interpolated_points):
    #     print(f"  Point {i}: lat={point['latitude']}, lon={point['longitude']}, alt={point['altitude']}")
    # print(f"Total interpolated points: {len(interpolated_points)}")
    # # print(f"Interpolated path points: {len(interpolated_points)}")

    # 平滑后的路径点列表
    smoothed_path = []

    # 添加第一个点
    first_point = interpolated_points[0].copy()
    first_point["curtime"] = 0.0
    smoothed_path.append(first_point)

    # 直接使用线性插值
    curtime = 0.0
    pathdict = 0.0
    p1 = interpolated_points[0]
    for i in range(len(interpolated_points) - 1):
        p2 = interpolated_points[i + 1]

        # 计算两点之间的距离
        dist = calculate_distance(
            p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
        )

        # 直线段上的点
        segment_time = dist / uav_speed
        num_points = max(1, int(segment_time / time_interval))
        if segment_time < time_interval:
            continue
        # 线性插值计算中间点
        for j in range(1, num_points + 1):
            fraction = j / (num_points + 1)
            # 线性插值计算中间点
            lat = p1["latitude"] + fraction * (p2["latitude"] - p1["latitude"])
            lon = p1["longitude"] + fraction * (p2["longitude"] - p1["longitude"])
            alt = p1["altitude"] + fraction * (p2["altitude"] - p1["altitude"])

            # 计算ENU坐标
            ref_lat = smoothed_path[0]["latitude"]
            ref_lon = smoothed_path[0]["longitude"]
            ref_alt = smoothed_path[0]["altitude"]
            east, north, up = lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

            # 计算smoothed_path中最后一个点和当前插值点之间的距离
            distance_to_last_point = 0.0
            if len(smoothed_path) > 0:
                last_point = smoothed_path[-1]
                distance_to_last_point = calculate_distance(
                    lat, lon, last_point["latitude"], last_point["longitude"]
                )
                # distance_to_last_point = math.sqrt(
                #     (east - last_point['east'])**2 +
                #     (north - last_point['north'])**2 +
                #     (up - last_point['up'])**2
                # )
                # if i == 0:
                #     print(f"distance_to_last_point1:{distance_to_last_point1},distance_to_last_point:{distance_to_last_point}")

            pathdict = pathdict + distance_to_last_point
            curtime = pathdict / uav_speed
            if i == 0:
                print(f"pathdict:{pathdict},curtime:{curtime}")
            smoothed_path.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "east": east,
                    "north": north,
                    "up": up,
                    "time": curtime,
                    "pathdict": pathdict,
                    "description": interpolated_points[i]["description"],
                }
            )

        p1 = p2

    # 添加最后一个点
    if len(interpolated_points) > 1:
        last_point = interpolated_points[-1].copy()

        distance_to_last_point = 0.0
        if len(smoothed_path) > 0:
            distance_to_last_point = math.sqrt(
                (east - last_point["east"]) ** 2
                + (north - last_point["north"]) ** 2
                + (up - last_point["up"]) ** 2
            )

        pathdict = pathdict + distance_to_last_point
        curtime = pathdict / uav_speed

        last_point["pathdict"] = pathdict
        last_point["time"] = curtime
        last_point["description"] = interpolated_points[-1]["description"]
        smoothed_path.append(last_point)

    return smoothed_path


def plot_smoothed_uav_path(
    original_path_points,
    smoothed_path_points,
    shooting_points=None,
    search_area_params=None,
):
    """
    绘制无人机平滑路径

    参数:
    - original_path_points: 原始路径点列表
    - smoothed_path_points: 平滑后的路径点列表
    - waypoints: 侦察路径点列表（可选，可以是螺旋或扫描路径）
    - shooting_points: 航拍点列表（可选）
    - search_area_params: 搜索区域参数（可选，包含center_east, center_north, radius）
    """
    # 提取原始路径的ENU坐标
    orig_east_coords = [point["east"] for point in original_path_points]
    orig_north_coords = [point["north"] for point in original_path_points]
    orig_up_coords = [point["up"] for point in original_path_points]

    # 提取平滑路径的ENU坐标
    smooth_east_coords = [point["east"] for point in smoothed_path_points]
    smooth_north_coords = [point["north"] for point in smoothed_path_points]
    smooth_up_coords = [point["up"] for point in smoothed_path_points]

    # 创建2D路径图
    plt.figure(figsize=(12, 5))

    # 子图1：水平面路径（东-北坐标）
    plt.subplot(1, 2, 1)
    plt.plot(
        orig_east_coords,
        orig_north_coords,
        "b--",
        linewidth=1,
        alpha=0.7,
        label="Original Path",
    )
    plt.plot(
        smooth_east_coords,
        smooth_north_coords,
        "g-",
        linewidth=1.5,
        label="Smoothed Path",
    )

    # 如果提供了搜索区域参数，绘制圆形区域和最小外接矩形
    if search_area_params:
        center_east = search_area_params["center_east"]
        center_north = search_area_params["center_north"]
        radius = search_area_params["radius"]

        # 绘制圆形区域
        circle = plt.Circle(
            (center_east, center_north),
            radius,
            color="gray",
            fill=False,
            linestyle="--",
            label="Search Area",
        )
        plt.gca().add_patch(circle)

        # 绘制最小外接矩形
        rect = plt.Rectangle(
            (center_east - radius, center_north - radius),
            2 * radius,
            2 * radius,
            fill=False,
            linestyle="-.",
            color="purple",
            label="Minimum Bounding Rectangle",
        )
        plt.gca().add_patch(rect)

        # 标注中心点
        plt.plot(center_east, center_north, "go", markersize=6, label="Search Center")

        # 标注矩形区域的顶点编号
        vertices = [
            (center_east - radius, center_north + radius),  # 左上角 (0)
            (center_east + radius, center_north + radius),  # 右上角 (1)
            (center_east + radius, center_north - radius),  # 右下角 (2)
            (center_east - radius, center_north - radius),  # 左下角 (3)
        ]

        for i, (x, y) in enumerate(vertices):
            plt.annotate(
                f"Vertex {i}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="purple",
            )

    # 如果有航拍点，也绘制
    if shooting_points:
        shooting_east = [point["east"] for point in shooting_points]
        shooting_north = [point["north"] for point in shooting_points]
        plt.scatter(
            shooting_east,
            shooting_north,
            color="orange",
            s=30,
            marker="x",
            label="Shooting Points",
        )

        # 标注前几个航拍点
        for i, (e, n) in enumerate(zip(shooting_east[:5], shooting_north[:5])):
            plt.annotate(
                f"Shooting Point {i+1}",
                (e, n),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                color="orange",
            )

    plt.xlabel("East Distance (m)")
    plt.ylabel("North Distance (m)")
    plt.title("UAV Smoothed Horizontal Path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # 确保坐标轴比例一致

    # 子图2：高度剖面图
    # 计算原始路径的累计距离
    orig_distances = []
    total_distance = 0
    orig_distances.append(total_distance)

    for i in range(1, len(original_path_points)):
        # 计算与前一个点的水平距离
        dx = orig_east_coords[i] - orig_east_coords[i - 1]
        dy = orig_north_coords[i] - orig_north_coords[i - 1]
        segment_distance = math.sqrt(dx**2 + dy**2)
        total_distance += segment_distance
        orig_distances.append(total_distance)

    # 计算平滑路径的累计距离
    smooth_distances = []
    total_distance = 0
    smooth_distances.append(total_distance)

    for i in range(1, len(smoothed_path_points)):
        # 计算与前一个点的水平距离
        dx = smooth_east_coords[i] - smooth_east_coords[i - 1]
        dy = smooth_north_coords[i] - smooth_north_coords[i - 1]
        segment_distance = math.sqrt(dx**2 + dy**2)
        total_distance += segment_distance
        smooth_distances.append(total_distance)

    plt.subplot(1, 2, 2)
    plt.plot(
        orig_distances,
        orig_up_coords,
        "b--",
        linewidth=1,
        alpha=0.7,
        label="Original Path",
    )
    plt.plot(
        smooth_distances, smooth_up_coords, "b-", linewidth=2, label="Smoothed Path"
    )

    # 如果有搜索区域信息，在高度图上标注
    if search_area_params:
        # 简化处理：在高度图上标注搜索区域中心点的投影位置
        pass

    # 如果有航拍点，也在高度图上标注
    if shooting_points:
        # 计算航拍点在高度图上的位置
        shooting_distances = []
        for point in shooting_points:
            # 找到航拍点在平滑路径上的最近点，并计算累计距离
            min_distance = float("inf")
            closest_index = 0
            for i, (east, north) in enumerate(
                zip(smooth_east_coords, smooth_north_coords)
            ):
                distance = math.sqrt(
                    (point["east"] - east) ** 2 + (point["north"] - north) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
            shooting_distances.append(smooth_distances[closest_index])

        plt.scatter(
            shooting_distances,
            [point["up"] for point in shooting_points],
            color="orange",
            s=30,
            marker="x",
            label="Shooting Points",
        )

    plt.xlabel("Cumulative Horizontal Distance (m)")
    plt.ylabel("Altitude (m)")
    plt.title("UAV Smoothed Altitude Profile")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("uav_smoothed_path.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_smoothed_waypoints_to_json(
    smoothed_scan_waypoints, photopoints, filename="uav_way.json"
):
    """
    将smoothed_scan_waypoints中的经纬高数据保存为JSON文件

    参数:
    smoothed_scan_waypoints: 包含路径点的列表，每个点包含经纬高数据
    filename: 保存的JSON文件名
    """

    # 提取经纬高数据
    waypoints_data = []
    for waypoint in smoothed_scan_waypoints:
        waypoint_info = {
            "latitude": waypoint.get("latitude", 0),
            "longitude": waypoint.get("longitude", 0),
            "altitude": waypoint.get("altitude", 0),
            "time": waypoint.get("time", 0),
        }
        waypoints_data.append(waypoint_info)

    waypoints_photo = []
    for waypoint in photopoints:
        waypoint_info = {
            "latitude": waypoint.get("latitude", 0),
            "longitude": waypoint.get("longitude", 0),
            "altitude": waypoint.get("altitude", 0),
            "time": waypoint.get("time", 0),
        }
        waypoints_photo.append(waypoint_info)

    searchstart = []
    for index, waypoint in enumerate(smoothed_scan_waypoints):
        if "search" in waypoint.get("description", ""):
            waypoint_info = {
                "latitude": waypoint.get("latitude", 0),
                "longitude": waypoint.get("longitude", 0),
                "altitude": waypoint.get("altitude", 0),
                "time": waypoint.get("time", 0),
                "Index": index,
            }
            searchstart.append(waypoint_info)
            break

    # 创建要保存的JSON数据
    data_to_save = {
        # "waypoints_count": len(waypoints_data),
        "waypoints": waypoints_data,
        "searchstart": searchstart,
        "photopoints": waypoints_photo,
    }

    # 保存到JSON文件
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"已保存 {len(waypoints_data)} 个路径点到 {filename}")

    return data_to_save


def save_search_area_params_to_json(search_area_params, filename="search_area.json"):
    """
    将search_area_params中的经纬高数据保存为JSON文件

    参数:
    search_area_params: 包含侦察区域顶点的列表，每个点包含经纬高数据
    filename: 保存的JSON文件名
    """
    # 提取经纬高数据
    search_area = []
    search_area_info = {
        "center_lat": search_area_params.get("center_lat", 0),
        "center_lon": search_area_params.get("center_lon", 0),
        "radius": search_area_params.get("radius", 0),
    }
    search_area.append(search_area_info)

    search_area_rect = []
    search_area_info = {
        "rect_lat1": search_area_params.get("rect_lat1", 0),
        "rect_lon1": search_area_params.get("rect_lon1", 0),
        "rect_lat2": search_area_params.get("rect_lat2", 0),
        "rect_lon2": search_area_params.get("rect_lon2", 0),
        "rect_lat3": search_area_params.get("rect_lat3", 0),
        "rect_lon3": search_area_params.get("rect_lon3", 0),
        "rect_lat4": search_area_params.get("rect_lat4", 0),
        "rect_lon5": search_area_params.get("rect_lon4", 0),
    }
    search_area_rect.append(search_area_info)

    # 创建要保存的JSON数据
    data_to_save = {
        "search_area_circle": search_area,
        "search_area_rect": search_area_rect,
    }

    # 保存到JSON文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"已保存 {len(search_area)} 个路径点到 {filename}")
