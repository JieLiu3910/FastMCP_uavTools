from collections import Counter

# 车辆类型中英文对照表
VEHICLE_LABEL_MAP = {
    "fire_truck": "消防车",
    "ambulance": "救护车",
    "police_car": "警车",
    "engineering_vehicle": "工程车",
    "truck": "卡车",
    "launch_vehicle": "发射车",
    "armored_vehicle": "装甲车",
    "tank": "坦克",
    "fuel_tanker": "油罐车",
    "bus": "大巴车",
    "command_vehicle": "指挥车",
    # "ship_055_999":"井冈山舰"
}

def get_chinese_label(english_label):
    """将英文标签转换为中文标签"""
    return VEHICLE_LABEL_MAP.get(english_label, english_label)
def analyze_target_label(results, top_n=10):
    """
    分析检索结果中前N个结果的label，返回判断结果
    """

    # 只分析前top_n个结果
    top_results = results[:top_n]

    labels = []
    for item in top_results:
        if 'label' in item:
            labels.append(item['label'])
    
    if labels:
        # 统计每个label出现的次数
        label_counts = Counter(labels)
        # 找出出现次数最多的label
        most_common_label, count = label_counts.most_common(1)[0] if label_counts else (None, 0)

        # 获取第一个结果的label和相似度得分
        first_result = top_results[0]
        first_label = first_result.get('label', 'N/A')
        first_similarity = first_result.get('distance', 0)  # 转换为相似度得分

        # 将相似度转换为百分比
        similarity_percentage = first_similarity * 100

        # 转换为中文标签
        chinese_most_common_label = get_chinese_label(most_common_label)
        chinese_first_label = get_chinese_label(first_label)
        chinese_labels = [get_chinese_label(label) for label in label_counts.keys()]

        # 构建详细的label分析结果
        label_analysis = {
            "top_n_analyzed": top_n,
            "label_statistics": dict(label_counts),
            "most_common_label": {
                "english": most_common_label,
                "chinese": chinese_most_common_label,
                "count": count,
                "percentage": (count / top_n) * 100
            },
            "top_result": {
                "english": first_label,
                "chinese": chinese_first_label,
                "similarity": first_similarity,
                "similarity_percentage": similarity_percentage
            },
            "conclusion": "",
            "confidence": "high" if most_common_label == first_label and count > 2 else "medium" if count > 2 else "low"
        }      
        
        # 判断条件：出现次数最多的label和排名第一的label一致
        if most_common_label == first_label:
                label_analysis["conclusion"] = f"目标图像与{chinese_most_common_label}高度契合，概率为{similarity_percentage:.2f}%，前{top_n}个检索结果中有{count}个同为该类别"
        elif count > 2:
            all_labels = "、".join([get_chinese_label(label) for label in label_counts.keys()])
            label_analysis["conclusion"] = f"目标图像疑似{all_labels}，与{chinese_first_label}最为相似，概率为{similarity_percentage:.2f}%"
        else:
            label_analysis["conclusion"] = "目标图像类别无法判断"
    else:
        label_analysis = {
            "conclusion": "目标图像未检索到任何有效标签信息",
            "confidence": "none"
        }
    
    return label_analysis