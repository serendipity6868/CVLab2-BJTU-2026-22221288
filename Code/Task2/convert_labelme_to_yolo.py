# -*- coding: gbk -*-
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

import os
import json
import shutil
import random
from PIL import Image

def convert_labelme_to_yolo():
    # 原始数据集路径
    src_data_dir = "BJTU-DataSet-2026/object_detection/data"
    
    # 目标 YOLO 数据集路径
    target_base = "Task2/yolo_dataset"
    modes = ['train', 'val']
    for mode in modes:
        os.makedirs(os.path.join(target_base, 'images', mode), exist_ok=True)
        os.makedirs(os.path.join(target_base, 'labels', mode), exist_ok=True)

    # 扫描所有的 json 文件
    all_jsons = [f for f in os.listdir(src_data_dir) if f.lower().endswith('.json')]
    
    # 随机打乱并按照 8:2 划分训练集和验证集
    random.seed(42)
    random.shuffle(all_jsons)
    split_idx = int(len(all_jsons) * 0.8)
    train_jsons = all_jsons[:split_idx]
    val_jsons = all_jsons[split_idx:]

    def process_split(json_list, mode):
        print(f"正在处理 {mode} 集合，共 {len(json_list)} 个样本...")
        for json_name in json_list:
            base_name = os.path.splitext(json_name)[0]
            json_path = os.path.join(src_data_dir, json_name)
            
            # 寻找对应的图片（支持 jpeg, jpg, png）
            img_name = None
            for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG']:
                if os.path.exists(os.path.join(src_data_dir, base_name + ext)):
                    img_name = base_name + ext
                    break
            
            if not img_name:
                continue
                
            # 读取图片宽和高用于归一化
            img_path = os.path.join(src_data_dir, img_name)
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # 读取 json 标签
            with open(json_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 创建对应的 YOLO txt 标签文件
            txt_path = os.path.join(target_base, 'labels', mode, base_name + '.txt')
            with open(txt_path, 'w', encoding='utf-8') as out_f:
                for shape in label_data.get('shapes', []):
                    # 统一当成文字区域目标，类别 ID 设为 0
                    class_id = 0 
                    points = shape['points'] # [[x1, y1], [x2, y2]]
                    
                    # 计算矩形边界框
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    xmin, xmax = min(x1, x2), max(x1, x2)
                    ymin, ymax = min(y1, y2), max(y1, y2)
                    
                    # 转换为 YOLO 格式: x_center, y_center, width, height (均归一化到 0-1)
                    w = xmax - xmin
                    h = ymax - ymin
                    x_center = xmin + w / 2.0
                    y_center = ymin + h / 2.0
                    
                    x_center_norm = x_center / img_w
                    y_center_norm = y_center / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    # 写入 txt 文件
                    out_f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            # 复制图片到指定目录
            shutil.copy(img_path, os.path.join(target_base, 'images', mode, img_name))

    process_split(train_jsons, 'train')
    process_split(val_jsons, 'val')
    print("数据格式化与划分完成！")

if __name__ == "__main__":
    convert_labelme_to_yolo()