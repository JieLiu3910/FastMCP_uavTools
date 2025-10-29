from fileinput import filename
import shutil
from ultralytics import YOLO
from pathlib import Path
import os
import glob
import json
import cv2
from collections import Counter
import yaml
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载配置文件
from config_manager import load_config


# classes_names = load_config()['yolo_label_names']
# print(f"classes_name: {classes_name}")

def predict(img_dir: str, res_dir: str = None)->dict:
    """
    predict the image

    Args:
        img_dir: the directory of the image (dir or image file <-- .tif/.png/.jpg)
        res_dir: the directory of the result (dir)

    Returns:
        dict: the path of the result json file and the path of the result image file

    Example:
        predict('./input/image/path/xxx.jpg','./output/directory')
        predict('./input/image/path','./output/directory')
    """

    weights = load_config()['yolo_weights']
    # Load a model
    model = YOLO(weights)  # pretrained YOLO11n model

    # get the image directory
    img_dir = Path(img_dir)
    if img_dir.is_dir():
        print(f"img_dir is: {img_dir}\n")
        images = [img_dir / img for img in img_dir.glob('*.jpg')]

    elif img_dir.is_file():
        print(f"img is : {img_dir}\n")
        images = [img_dir]
    else:
        print(f"img_dir is not a directory or file: {img_dir}\n")
        return

    
    # get the result directory
    if res_dir is None:
        res_dir = Path(__file__).parent.parent / "results"
        if not res_dir.exists():
            os.makedirs(res_dir)
            print(f"res_dir is: {res_dir}\n")
    else:
        res_dir = Path(res_dir)
        if not res_dir.exists():
            os.makedirs(res_dir)
            print(f"res_dir is: {res_dir}\n")
        
    # execute predict
    results = model.predict(images,iou=0.4,conf=0.6)  # return a list of Results objects

    out_info_list = []

    # Process results list
    for (result,image) in zip(results,images):
        # result.boxes: Boxes object for bounding box outputs
        # 获取所有检测到的框的类别索引

        # 获取所有检测到的框
        boxes = result.boxes
        # print(f"result.boxes: {boxes}")

        # 获取类别名称映射
        classes_names = result.names

        # 1.框的类别
        box_classes = boxes.cls.cpu().tolist()
        # print(f"box_classes: {box_classes}")

       # 2.框的类别名称
        box_class_names = [classes_names[int(c)] for c in box_classes]
        print(f"box_class_names: {box_class_names}")    

        # 3.框的坐标
        box_xyxy = boxes.xyxy.cpu().tolist()
        # print(f"box_xyxy: {box_xyxy}")

        # 4.框的置信度
        box_confidences = boxes.conf.cpu().tolist()
        # print(f"box_confidences: {box_confidences}")

        
        # 统计数量
        class_counts = Counter(box_class_names)              # 每类数量
                
        class_counts['total'] = sum(class_counts.values())   # 总数
        # print(f"目标种类和数量：{class_counts}") 


        # 6.框的坐标和置信度
        box_info = []
        for box, cls, conf in zip(box_xyxy, box_class_names, box_confidences):
            box_info.append({
                "class": cls,
                "confidence": conf,
                "bbox": box,
            })
        print(f"\nbox_info: {box_info}\n")

        # result.show()   # display to screen

        # save predict image
        predict_dir = res_dir / "predicts"
        if not predict_dir.exists():
            os.makedirs(predict_dir)
            print(f"predict_dir is: {predict_dir}\n")
        
        save_img_name =  predict_dir / f"{image.name.split('.')[0]}_predict.jpg"


        annotated_img = result.plot(line_width=2, font_size=20,save=True,   # save=True 保存图片
            filename=save_img_name,txt_color=(255,255,255),labels=False)

        # result.save(filename = save_img)  # save
        # cv2.imwrite(str(save_img), annotated_img)
        print(f"save image to {save_img_name}\n")
       
        # save predict info
        
        out_info = {
            "image_name":image.name,
            "image_path":str(image.parent),
            "counts":class_counts,
            "detection":box_info,
        }
        # out_info_list.append(out_info)

        # 保存预测信息.json文件
        save_json_name = predict_dir / f"{image.name.split('.')[0]}_predict.json"
        with open(save_json_name, 'w') as f:
            json.dump(out_info, f, indent=4)

        # 复制原始图像到路径
        raw_img_dir = res_dir / "photographs"
        if not raw_img_dir.exists():
            os.makedirs(raw_img_dir)
            print(f"raw_img_dir is: {raw_img_dir}\n")
        
        # 如果原始图像不在photographs目录下，则复制到photographs目录下
        if image.parent != raw_img_dir:
            shutil.copy(image, raw_img_dir)

    return {
        "filename":f"{image.name.split('.')[0]}_predict.jpg",
        "image_path":str(image.parent),
        "objects_counts":class_counts,
        "predicted_json_path": str(save_json_name), 
        "predicted_image_path": str(save_img_name)
        }  


if __name__ == "__main__":

    img_dir = r"D:\\04-Code\\Learn\\FastMCP_demo\\results\\photographs\\LX3-1-1_02.jpg" # 使用绝对路径
    # res_dir = "/results"
    predict(img_dir)