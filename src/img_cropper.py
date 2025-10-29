#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标裁剪脚本
根据目标检测结果JSON文件或YOLO格式TXT文件，从原始图像中裁剪出检测到的目标并保存为独立的JPG文件
"""

import json
import os
import argparse
from PIL import Image
import sys
import requests
import io
from pathlib import Path

import yaml
from urllib.parse import urlparse


def is_url(path):
    """
    检查给定路径是否为网络 URL
    
    Args:
        path (str): 要检查的路径或URL字符串
        
    Returns:
        bool: 如果是URL则返回True，否则返回False
    """
    return path.startswith('http://') or path.startswith('https://')


def parse_predic_info(json_file_path):
    """
    解析检测结果JSON文件（支持本地文件和URL）
    
    Args:
        json_file_path (str): JSON文件的路径或URL
        
    Returns:
        tuple: (image_path, image_name, detection) 原始图像路径、文件名和检测结果列表
        
    Raises:
        FileNotFoundError: 当JSON文件不存在时
        json.JSONDecodeError: 当JSON格式错误时
        KeyError: 当JSON文件缺少必要字段时
        requests.exceptions.RequestException: 当网络请求失败时
    """
    try:
        # 判断是URL还是本地路径
        if is_url(json_file_path):
            # 从URL加载JSON数据
            print(f"从URL加载JSON文件: {json_file_path}")
            
            try:
                response = requests.get(json_file_path, timeout=30)
                response.raise_for_status()  # 检查请求状态码
                data = response.json()  # 直接解析JSON响应
                print("成功从URL加载JSON数据")
            except requests.exceptions.RequestException as e:
                print(f"错误: 从URL加载JSON失败 - {e}")
                raise
            except json.JSONDecodeError as e:
                print(f"错误: URL返回的内容不是有效的JSON格式 - {e}")
                raise
        else:
            # 从本地文件加载JSON数据
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功解析本地JSON文件: {json_file_path}")
        
        # 提取关键信息
        image_path = data['image_path']
        image_name = data['image_name']
        detection = data['detection']
        
        # 验证detection列表中的每个元素是否包含必要字段
        for i, det in enumerate(detection):
            if 'class' not in det or 'bbox' not in det:
                raise KeyError(f"检测结果第{i+1}项缺少'class'或'bbox'字段")
        
        print(f"图像路径: {image_path}")
        print(f"图像文件名: {image_name}")
        print(f"检测到 {len(detection)} 个目标")
        
        return image_path, image_name, detection
        
    except FileNotFoundError:
        print(f"错误: 找不到JSON文件 '{json_file_path}'")
        raise
    except requests.exceptions.RequestException as e:
        print(f"错误: 网络请求失败 - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
        raise
    except KeyError as e:
        print(f"错误: JSON文件缺少必要字段 - {e}")
        raise
    except Exception as e:
        print(f"错误: 解析JSON文件时发生未知错误 - {e}")
        raise


def parse_yolo_label_info(label_file_path, classes=None):
    """
    解析YOLO格式的TXT标签文件（支持本地文件和URL）
    
    Args:
        label_file_path (str): YOLO格式的.txt标签文件路径或URL
        classes (list, optional): 类别名称列表，用于将类别ID映射到具体名称。
                                 例如: ["ambulance", "bus", "car"]，则ID 0->ambulance, 1->bus, 2->car
                                 如果为None，则直接使用类别ID作为类别名称
        
    Returns:
        tuple: (image_path, image_name, detections) 原始图像路径、文件名和检测结果列表
        
    Raises:
        FileNotFoundError: 当标签文件或对应图像文件不存在时
        ValueError: 当标签文件格式错误时
        requests.exceptions.RequestException: 当网络请求失败时
    """
    try:
        # 判断是URL还是本地路径
        if is_url(label_file_path):
            # 从URL处理YOLO标签文件
            print(f"从URL加载YOLO标签文件: {label_file_path}")
            
            try:
                # 获取标签文件内容
                response = requests.get(label_file_path, timeout=30)
                response.raise_for_status()
                label_content = response.text
                print("成功从URL加载YOLO标签数据")
                
                # 从URL推断图像URL
                image_url = None
                image_name = None
                
                # 获取标签文件的基础文件名（不含扩展名）
                from urllib.parse import urlparse
                parsed_url = urlparse(label_file_path)
                label_basename = os.path.splitext(os.path.basename(parsed_url.path))[0]
                
                # 尝试不同的图像扩展名
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                
                # 如果URL路径中包含'labels'，尝试替换为'images'
                if 'labels' in label_file_path:
                    base_image_url = label_file_path.replace('labels', 'images')
                    for ext in image_extensions:
                        test_image_url = base_image_url.replace('.txt', ext)
                        try:
                            test_response = requests.head(test_image_url, timeout=10)
                            if test_response.status_code == 200:
                                image_url = test_image_url
                                image_name = os.path.basename(urlparse(test_image_url).path)
                                print(f"找到对应图像URL: {image_url}")
                                break
                        except requests.exceptions.RequestException:
                            continue
                
                # 如果没找到图像URL，抛出错误
                if image_url is None:
                    raise FileNotFoundError(f"找不到与标签文件URL '{label_file_path}' 对应的图像文件")
                
                # 从URL加载图像以获取尺寸
                try:
                    image_response = requests.get(image_url, timeout=30)
                    image_response.raise_for_status()
                    
                    # 使用内存中的图像数据获取尺寸
                    with Image.open(io.BytesIO(image_response.content)) as img:
                        img_width, img_height = img.size
                    print(f"从URL加载图像并获取尺寸: {img_width} x {img_height}")
                except Exception as e:
                    raise ValueError(f"无法从URL加载图像文件 '{image_url}': {e}")
                
                # 标签内容按行分割处理
                label_lines = label_content.strip().split('\n')
                image_path = image_url  # 对于URL，直接返回完整的图像URL作为路径
                
            except requests.exceptions.RequestException as e:
                print(f"错误: 从URL加载标签文件失败 - {e}")
                raise
                
        else:
            # 处理本地文件（保持原有逻辑）
            # 检查标签文件是否存在
            if not os.path.exists(label_file_path):
                raise FileNotFoundError(f"标签文件不存在: {label_file_path}")
            
            # 获取标签文件的目录和基础文件名
            label_dir = os.path.dirname(label_file_path)
            label_basename = os.path.splitext(os.path.basename(label_file_path))[0]
            
            # 查找对应的图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            image_path = None
            image_name = None
            
            # 方式一：检查是否为标准YOLO数据集结构 (labels目录 -> images目录)
            if 'labels' in label_dir:
                # 将labels目录替换为images目录
                images_dir = label_dir.replace('labels', 'images')
                
                print(f"检测到YOLO数据集结构，查找图像目录: {images_dir}")
                
                for ext in image_extensions:
                    potential_image_path = os.path.join(images_dir, label_basename + ext)
                    if os.path.exists(potential_image_path):
                        image_path = images_dir
                        image_name = label_basename + ext
                        print(f"找到对应图像文件: {potential_image_path}")
                        break
            
            # 如果都没找到，抛出错误
            if image_path is None:
                raise FileNotFoundError(f"找不到与标签文件 '{label_file_path}' 对应的图像文件\n"
                                      f"已尝试查找路径：\n"
                                      f"1. YOLO数据集结构: {label_dir.replace('labels', 'images') if 'labels' in label_dir else 'N/A'}")
            
            # 使用PIL加载图像以获取尺寸
            full_image_path = os.path.join(image_path, image_name)
            try:
                with Image.open(full_image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                raise ValueError(f"无法加载图像文件 '{full_image_path}': {e}")
            
            # 读取本地标签文件
            with open(label_file_path, 'r', encoding='utf-8') as f:
                label_lines = f.readlines()
        
        # 显示类别映射信息
        if classes is not None:
            print(f"使用类别映射: {len(classes)} 个类别")
            print(f"类别列表: {classes}")
        else:
            print("未提供类别映射，将使用类别ID作为类别名称")
        
        # 解析YOLO标签数据
        detections = []
        for line_num, line in enumerate(label_lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            try:
                # 解析YOLO格式: class_id center_x center_y width height
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"第{line_num}行格式错误，应为5个数值")
                
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 验证归一化坐标范围
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                       0 <= width <= 1 and 0 <= height <= 1):
                    print(f"警告: 第{line_num}行坐标超出归一化范围 [0,1]")
                
                # 将归一化坐标转换为绝对像素坐标
                # YOLO格式: (center_x, center_y, width, height) -> (xmin, ymin, xmax, ymax)
                xmin = (center_x - width / 2) * img_width
                ymin = (center_y - height / 2) * img_height
                xmax = (center_x + width / 2) * img_width
                ymax = (center_y + height / 2) * img_height
                
                # 确保坐标在图像范围内
                xmin = max(0, min(img_width, xmin))
                ymin = max(0, min(img_height, ymin))
                xmax = max(0, min(img_width, xmax))
                ymax = max(0, min(img_height, ymax))
                
                # 进行类别映射
                if classes is not None and 0 <= class_id < len(classes):
                    # 使用映射后的类别名称
                    class_name = classes[class_id]
                elif classes is not None:
                    # 类别ID超出范围，给出警告并使用原ID
                    print(f"警告: 第{line_num}行类别ID {class_id} 超出类别列表范围 (0-{len(classes)-1})，使用原ID")
                    class_name = str(class_id)
                else:
                    # 没有提供类别映射，直接使用类别ID
                    class_name = str(class_id)
                
                # 构建与JSON格式兼容的检测结果
                detection = {
                    'class': class_name,  # 使用映射后的类别名称或原ID
                    'bbox': [xmin, ymin, xmax, ymax]
                }
                detections.append(detection)
                
            except ValueError as e:
                print(f"警告: 第{line_num}行解析错误 - {e}")
                continue
        
        print(f"成功解析YOLO标签文件: {label_file_path}")
        if is_url(label_file_path):
            print(f"对应图像: {image_url}")
        else:
            print(f"对应图像: {os.path.join(image_path, image_name)}")
        print(f"图像尺寸: {img_width} x {img_height}")
        print(f"检测到 {len(detections)} 个目标")
        
        return image_path, image_name, detections
        
    except FileNotFoundError:
        print(f"错误: 文件未找到")
        raise
    except requests.exceptions.RequestException as e:
        print(f"错误: 网络请求失败 - {e}")
        raise
    except Exception as e:
        print(f"错误: 解析YOLO标签文件时发生未知错误 - {e}")
        raise


def crop_and_save_objects(image_path: str, image_name: str, detections: list, output_dir: str)->list:
    """
    裁剪目标并保存为独立图像文件（支持本地文件和URL）
    
    Args:
        image_path (str): 原始图像所在的目录路径或完整的URL
        image_name (str): 原始图像的文件名
        detections (list): 检测目标信息列表
        output_dir (str): 输出目录路径
        
    Returns:
        int: 成功裁剪保存的目标数量
    """
    try:
        # 判断是URL还是本地路径
        if is_url(image_path):
            # 对于URL情况，image_path就是完整的图像URL
            full_image_path = image_path
            print(f"从URL加载图像: {full_image_path}")
            
            # 从URL加载图像
            try:
                response = requests.get(full_image_path, timeout=30)
                response.raise_for_status()
                original_image = Image.open(io.BytesIO(response.content))
                print(f"成功从URL加载原始图像")
                print(f"图像尺寸: {original_image.size[0]} x {original_image.size[1]}")
            except requests.exceptions.RequestException as e:
                print(f"错误: 无法从URL加载图像文件 '{full_image_path}' - {e}")
                return 0
            except Exception as e:
                print(f"错误: 无法解析URL图像数据 '{full_image_path}' - {e}")
                return 0
                
        else:
            # 处理本地文件
            # 拼接完整的原始图像路径
            full_image_path = os.path.join(image_path, image_name)
            
            # 检查原始图像是否存在
            if not os.path.exists(full_image_path):
                print(f"❌️错误: 找不到原始图像文件 '{full_image_path}'")
                return 0
            
            # 加载原始图像
            try:
                original_image = Image.open(full_image_path)
                print(f"🖼️成功加载本地原始图像: {full_image_path}")
                print(f"📏图像尺寸: {original_image.size[0]} x {original_image.size[1]}")
            except Exception as e:
                print(f"❌️错误: 无法加载图像文件 '{full_image_path}' - {e}")
                return 0
        
        # 检查并创建输出目录
        output_dir = os.path.abspath(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📂创建输出目录: {output_dir}")
        
        # 获取图像文件名（不含扩展名）
        image_basename = os.path.splitext(image_name)[0]
        
        # 统计每个类别的数量，用于生成唯一文件名
        class_counts = {}
        output_path_list = []
  
        
        # 遍历所有检测目标
        for i, detection in enumerate(detections):
            try:
                # 获取类别和边界框坐标
                obj_class = detection['class']
                bbox = detection['bbox']  # [xmin, ymin, xmax, ymax]
                
                # 统计类别数量
                if obj_class not in class_counts:
                    class_counts[obj_class] = 0
                class_counts[obj_class] += 1
                
                # 确保坐标为整数并在有效范围内
                xmin = max(0, int(bbox[0]))
                ymin = max(0, int(bbox[1]))
                xmax = min(original_image.size[0], int(bbox[2]))
                ymax = min(original_image.size[1], int(bbox[3]))
                
                # 验证边界框的有效性
                if xmin >= xmax or ymin >= ymax:
                    print(f"警告: 第{i+1}个目标的边界框无效 ({xmin}, {ymin}, {xmax}, {ymax})，跳过")
                    continue
                
                # 裁剪目标区域
                cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
                
                # 如果图像是RGBA模式，转换为RGB模式以便保存为JPEG
                if cropped_image.mode == 'RGBA':
                    # 创建白色背景
                    rgb_image = Image.new('RGB', cropped_image.size, (255, 255, 255))
                    # 将RGBA图像粘贴到白色背景上
                    rgb_image.paste(cropped_image, mask=cropped_image.split()[-1])  # 使用alpha通道作为mask
                    cropped_image = rgb_image
                elif cropped_image.mode != 'RGB':
                    # 其他模式也转换为RGB
                    cropped_image = cropped_image.convert('RGB')
                
                # 生成文件名: {image_name}_{class}_{index}.jpg
                output_filename = f"{image_basename}_{obj_class}_{class_counts[obj_class]}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存裁剪的图像
                cropped_image.save(output_path, 'JPEG', quality=95)
                
                output_path_list.append(output_path)
                
                print(f"成功裁剪保存: {output_path} (尺寸: {cropped_image.size[0]}x{cropped_image.size[1]})")
                
            except Exception as e:
                print(f"错误: 处理第{i+1}个目标时发生错误 - {e}")
                continue
        
        # 关闭原始图像
        original_image.close()
        
        return output_path_list
        
    except Exception as e:
        print(f"错误: 裁剪和保存过程中发生未知错误 - {e}")
        return 1


def main(input_label_path: str = None, out_dir: str = None, classes: list = None)->list:
    """
    主执行函数
    Args:
        input_label_path: 输入的标签json文件路径
        out_dir: 输出目录路径
        classes: 类别列表
    Returns:
        cropped_results: 裁剪后的图像路径列表
    """
    
    # 参数优先级：直接传入的参数 > 默认值
    if out_dir is not None:
        output_dir = out_dir
    else:
        output_dir = Path(__file__).parent.parent / 'results' / 'objects'
        
    if classes is not None:
        final_classes = classes
    else:                         # 若classes参数缺失，默认分类表
        final_classes = [
            "ambulance", 
            "armored_vehicle", 
            "bus", 
            "command_vehicle", 
            "engineering_vehicle", 
            "fire_truck",
            "fuel_tanker",
            "launch_vehicle",
            "police_car",
            "tank",
            "truck"
        ]

    
    # 显示类别映射信息
    print(f"  类别映射: {len(final_classes)} 个类别 {final_classes}\n")
    print("=" * 60)
    
    try:
        # 检查输入文件是否存在（仅对本地文件检查）
        if not is_url(input_label_path) and not os.path.exists(input_label_path):
            print(f"错误: 输入文件不存在 '{input_label_path}'")
            sys.exit(1)
        
        # 根据文件扩展名选择解析方法
        if is_url(input_label_path):
            # 对于URL，从路径中提取扩展名
            parsed_url = urlparse(input_label_path)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
        else:
            file_extension = os.path.splitext(input_label_path)[1].lower()
        
        if file_extension == '.json':
            # 步骤1: 解析JSON文件
            print("步骤1: 解析JSON文件...")
            image_path, image_name, detections = parse_predic_info(input_label_path)
        elif file_extension == '.txt':
            # 步骤1: 解析YOLO TXT文件
            print("步骤1: 解析YOLO TXT标签文件...")
            image_path, image_name, detections = parse_yolo_label_info(input_label_path, final_classes)
        else:
            print(f"错误: 不支持的文件类型 '{file_extension}'")
            print("支持的文件类型: .json（检测结果）, .txt（YOLO标签）")
            sys.exit(1)
        
        print("-" * 60)
        
        # 步骤2: 裁剪目标并保存
        print("步骤2: 裁剪目标并保存...")
        cropped_results = crop_and_save_objects(image_path, image_name, detections, output_dir)
        
        print("-" * 60)
        print("处理完成!")
        print(f"🔍️总共检测到 {len(detections)} 个目标")
        print(f"✅️成功裁剪保存 {len(cropped_results)} 个目标")
        print(f"📂文件保存位置: {os.path.abspath(output_dir)}")
        
        if len(cropped_results) < len(detections):
            print(f"❗️注意: 有 {len(detections) - len(cropped_results)} 个目标未能成功处理")
            
    except KeyboardInterrupt:
        print("\n❌️用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌️程序执行失败: {e}")
        sys.exit(1)
    
    print("=" * 60)

    return cropped_results


if __name__ == '__main__':

    # 没有命令行参数，使用预设参数直接调用
    label_path =  r'D:\04-Code\Learn\FastMCP_demo\results\predicts\group3-1(12)_predict_info.json' # URL示例 （必选）
    main(label_path) 
