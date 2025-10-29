import json
import math
from datetime import datetime
import os
from pprint import pprint
from typing import List, Dict
from collections import defaultdict

from torch.distributed import all_reduce

class VehicleRouteAnalyzer:
    def __init__(self, history_results: List[Dict] = None, json_file_path: str = None, 
                 max_speed_kmh: float = 120, time_threshold: int = 36000, 
                 distance_threshold: float = 100000):
        """
        初始化车辆路线分析器
        
        Args:
            history_results: 历史结果数据列表（JSON类型变量）
            json_file_path: JSON文件路径（为了向后兼容）
            json_file_path: JSON文件路径
            max_speed_kmh: 最大合理速度(km/h)
            time_threshold: 时间阈值(秒)，用于判断是否属于同一轨迹
            distance_threshold: 距离阈值(米)，用于空间聚类
        """
        self.max_speed_ms = max_speed_kmh * 1000 / 3600
        self.time_threshold = time_threshold
        self.distance_threshold = distance_threshold

        # 优先使用传入的history_results数据，否则从文件加载
        if history_results is not None:
            self.raw_data = history_results
        elif json_file_path is not None:
            self.raw_data = self.load_data_from_file(json_file_path)
        else:
            raise ValueError("必须提供history_results或json_file_path参数")

        self.data = self.filter_duplicate_images()
        self.vehicles = self.group_by_spatiotemporal_clustering()
    
    def load_data_from_file(self, json_file_path: str) -> List[Dict]:
        """从文件加载JSON数据（为了向后兼容）"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def filter_duplicate_images(self) -> List[Dict]:
        """
        筛选和合并相同image_id的数据
        对于相同image_id的数据，只保留其中一个的时间戳和经纬度信息
        """
        image_groups = defaultdict(list)
        
        # 按image_id分组
        for item in self.raw_data:
            image_groups[item['image_id']].append(item)
        
        filtered_data = []
        
        for image_id, items in image_groups.items():
            if len(items) == 1:
                filtered_data.append(items[0])
            else:
                # 选择时间戳最早的一个
                selected_item = min(items, key=lambda x: x['timestamp'])
                filtered_data.append(selected_item)
        
        print(f"数据筛选完成: 原始数据{len(self.raw_data)}条，去重后{len(filtered_data)}条")
        return filtered_data
    
    def group_by_spatiotemporal_clustering(self) -> Dict[str, List[Dict]]:
        """
        基于时空特征的车辆分组
        使用时间和空间信息来识别可能的同一车辆轨迹
        """
        # 首先按时间排序所有数据点
        sorted_data = sorted(self.data, key=lambda x: x['timestamp'])
        
        vehicles = {}
        current_vehicle_id = 0
        
        for i, point in enumerate(sorted_data):
            assigned = False
            
            # 尝试将点分配到现有车辆轨迹
            for vehicle_id, points in vehicles.items():
                last_point = points[-1]
                
                # 计算与上一个点的时间和空间距离
                time_diff = point['timestamp'] - last_point['timestamp']
                distance = self.haversine_distance(
                    point['latitude'], point['longitude'],
                    last_point['latitude'], last_point['longitude']
                )
                
                # 如果时间和空间距离合理，认为是同一车辆
                if (time_diff > 0 and time_diff < self.time_threshold and 
                    distance < self.distance_threshold):
                    points.append(point)
                    assigned = True
                    break
            
            # 如果没有分配到现有车辆，创建新车辆
            if not assigned:
                vehicle_id = f"vehicle_{current_vehicle_id}"
                vehicles[vehicle_id] = [point]
                current_vehicle_id += 1
        
        # 过滤掉2个点及以下的轨迹
        vehicles = {vid: points for vid, points in vehicles.items() if len(points) > 2}
        
        print(f"时空聚类识别出 {len(vehicles)} 条可能车辆轨迹")
        return vehicles
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两个经纬度坐标之间的球面距离（米）"""
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi/2) * math.sin(delta_phi/2) +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda/2) * math.sin(delta_lambda/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def calculate_speed(self, point1: Dict, point2: Dict) -> float:
        """计算两点之间的平均速度（米/秒）"""
        distance = self.haversine_distance(
            point1['latitude'], point1['longitude'],
            point2['latitude'], point2['longitude']
        )
        time_diff = abs(point2['timestamp'] - point1['timestamp'])
        
        if time_diff == 0:
            return float('inf')
        
        return distance / time_diff
    
    def is_route_possible(self, points: List[Dict]) -> bool:
        """判断路线是否可能（基于速度合理性）"""
        if len(points) < 2:
            return False
        
        for i in range(len(points) - 1):
            speed = self.calculate_speed(points[i], points[i + 1])
            if speed > self.max_speed_ms:
                print(f"轨迹速度超标: {speed:.2f} m/s (允许: {self.max_speed_ms:.2f} m/s)")
                return False
        
        return True
    
    def analyze_routes(self) -> List[Dict]:
        """分析所有可能的路线"""
        possible_routes = []
        
        for vehicle_id, points in self.vehicles.items():
            if self.is_route_possible(points):
                start_point = points[0]
                end_point = points[-1]
                
                # 计算总距离和总时间
                total_distance = 0
                total_time = end_point['timestamp'] - start_point['timestamp']
                
                for i in range(len(points) - 1):
                    total_distance += self.haversine_distance(
                        points[i]['latitude'], points[i]['longitude'],
                        points[i+1]['latitude'], points[i+1]['longitude']
                    )
                
                # 提取所有点的经纬度信息
                latitudes = [point['latitude'] for point in points]
                longitudes = [point['longitude'] for point in points]
                timestamps = [point['timestamp'] for point in points]
                image_ids = [point['image_id'] for point in points]

                route_info = {
                    'route_id': vehicle_id,
                    'start_timestamp': start_point['timestamp'],
                    'start_latitude': start_point['latitude'],
                    'start_longitude': start_point['longitude'],
                    'end_timestamp': end_point['timestamp'],
                    'end_latitude': end_point['latitude'],
                    'end_longitude': end_point['longitude'],
                    'num_points': len(points),
                    'total_distance_km': total_distance / 1000,
                    'total_time_hours': total_time / 3600,
                    'average_speed_kmh': (total_distance / 1000) / (total_time / 3600) if total_time > 0 else 0,
                    # 添加所有路线点的详细信息
                    'points': {
                        'latitude': latitudes,
                        'longitude': longitudes,
                        'timestamp': timestamps,
                        'image_id': image_ids
                    }
                }
                possible_routes.append(route_info)
        
        return possible_routes
    
    def save_routes(self, routes: List[Dict], output_file: str):
        """保存路线信息到JSON文件"""
        for route in routes:
            route['start_datetime'] = datetime.fromtimestamp(route['start_timestamp']).isoformat()
            route['end_datetime'] = datetime.fromtimestamp(route['end_timestamp']).isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(routes, f, indent=2, ensure_ascii=False)
        
        print(f"已保存 {len(routes)} 条路线到 {output_file}")



def remove_duplicate_routes(routes):
    """
    严格的去重：只有当起点和终点都相同时才认为是重复轨迹
    """
    if not routes:
        return routes
    
    unique_routes = []
    seen_route_pairs = set()
    
    for route in routes:
        # 提取起点和终点坐标并保留小数点后三位
        start_lat = round(route['start_latitude'], 3)
        start_lon = round(route['start_longitude'], 3)
        end_lat = round(route['end_latitude'], 3)
        end_lon = round(route['end_longitude'], 3)
        
        # 创建轨迹标识符（起点+终点）
        route_key = (start_lat, start_lon, end_lat, end_lon)
        
        # 检查是否已经存在相同的轨迹
        if route_key not in seen_route_pairs:
            unique_routes.append(route)
            seen_route_pairs.add(route_key)
        else:
            # 跳过重复轨迹
            continue
    
    return unique_routes

def analyze_route_main(json_file_list: List[str]):
    # 初始化分析器（阈值参数也可以调整，不给就用默认值）
    all_results = {}
    all_routes = []
    all_json_file_names = []
    
    for json_file in json_file_list:
        json_file_name = os.path.basename(json_file)
        print(f"\n{"="*100}")
        print(f"开始分析 {json_file}")
        analyzer = VehicleRouteAnalyzer(json_file_path=json_file)
        
        # 使用规则分析路线
        print("进行基于规则的路线分析...")
        rule_based_routes = analyzer.analyze_routes()
        print(f"基于规则分析发现 {len(rule_based_routes)} 条可能路线")

        # 合并结果（去重）
        # 在保存结果时使用
        if rule_based_routes:
            # 去除重复轨迹
            unique_routes = remove_duplicate_routes(rule_based_routes)
       
            # 打印结果摘要
            print("\n可能的车辆行驶路线：")
            for i, route in enumerate(unique_routes, 1):
                print(f"{i}. 轨迹 {route['route_id']}")
                print(f"   起点: {route['start_latitude']:.6f}, {route['start_longitude']:.6f}")
                print(f"   终点: {route['end_latitude']:.6f}, {route['end_longitude']:.6f}")
                print(f"   时间: {route['start_timestamp']} -> {route['end_timestamp']}")
                if 'total_distance_km' in route:
                    print(f"   距离: {route['total_distance_km']:.2f} km")
                    print(f"   平均速度: {route['average_speed_kmh']:.2f} km/h")
                print(f"   轨迹点数: {route['num_points']}")
                if 'analysis_method' in route:
                    print(f"   分析方法: {route['analysis_method']}")
                    print(f"   置信度: {route.get('llm_confidence', 'N/A')}")
        else:
            print("未发现可能的行驶路线")

        all_routes.append(unique_routes)
        all_json_file_names.append(json_file_name)

    # all_results["json_file_name"] = all_json_file_names
    # all_results["data"] = all_routes
    # all_results["data_count"] = len(all_routes)
    # all_results["type"] = "vehicle_route"

    return all_routes

 
if __name__ == "__main__":
    json_file_list = [
        "D:\\04-Code\\Learn\\FastMCP_demo\\results\\history_search\\LX5-1-5_03_armored_vehicle_1_history.json",
        "D:\\04-Code\\Learn\\FastMCP_demo\\results\\history_search\\LX5-1-5_03_tank_1_history.json"]

    all_routes = analyze_route_main(json_file_list)
    print(f"\n{"="*100}")
    print(f"所有可能的行驶路线:")
    pprint(all_routes)