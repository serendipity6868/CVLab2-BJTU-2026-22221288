# -*- coding: gbk -*-
import os
import sys
import io
import json
import random
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. 忽略 Python 的标准告警提示
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=UserWarning)

# 2. 防止 Windows 终端下的 GBK 打印报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 3. 增强中文字体后备列表，防止 Matplotlib 渲染标题时出现方格乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torchvision

# ==================== 纯 Python 实现的 NMS 替换补丁 ====================
def python_nms(boxes, scores, iou_threshold):
    """
    用纯 PyTorch 算子实现的 NMS 算法，不依赖底层 C++ 拓展。
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0].item()
        keep.append(i)
        
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = torch.where(ovr <= iou_threshold)[0]
        order = order[ids + 1]
        
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

# 【关键置换】用我们自己写的免编译 Python NMS 强制顶替掉官方那个会崩溃的 C++ NMS
torchvision.ops.nms = python_nms
# ===================================================================================

from ultralytics import YOLO

def load_ground_truth_boxes(json_path):
    """读取原本 labelme 格式 json 文件中的真实矩形框坐标"""
    boxes = []
    if not os.path.exists(json_path):
        return boxes
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            points = shape['points']  # [[x1, y1], [x2, y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            boxes.append([xmin, ymin, xmax, ymax])
    except Exception:
        pass
    return boxes

def run_joint_evaluation():
    # 路径配置
    stats_json_path = "BJTU-DataSet-2026/object_detection/label_statistics.json"
    data_dir = "BJTU-DataSet-2026/object_detection/data"
    weight_path = "Task2/runs/text_detection/weights/best.pt"
    output_dir = "Task2/visualize_results"
    
    # 检查核心文件是否存在
    if not os.path.exists(stats_json_path):
        print(f"未找到统计文件：{stats_json_path}")
        return
    if not os.path.exists(weight_path):
        print(f"未找到最优训练权重：{weight_path}，请确认 Task2 训练生成的序号。")
        return

    # 加载最优 YOLO 模型
    model = YOLO(weight_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取地标统计数据
    with open(stats_json_path, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)
    
    # 挑选出符合规范的所有地标类别（包含文件提供的 24 个大类）
    all_categories = stats_data.get('label_statistics', {})
    valid_landmarks = sorted(list(all_categories.keys()))
        
    print(f"成功解析数据集！将对以下 {len(valid_landmarks)} 个地标类别进行对比验证：")
    print(valid_landmarks)
    print("-" * 70)

    random.seed(42)  # 固定随机种子，确保每次运行挑选的数据一致
    total_groups = 0

    # 遍历每一个地标类别
    for landmark in valid_landmarks:
        files_list = all_categories[landmark].get('files', [])
        # 如果当前大类的样本数不足 2 个，则跳过或直接拿取全部
        if len(files_list) == 0:
            continue
        elif len(files_list) == 1:
            selected_jsons = files_list * 2 # 复制一下凑够两组
        else:
            selected_jsons = random.sample(files_list, 2)
        
        for idx, json_name in enumerate(selected_jsons):
            base_name = os.path.splitext(json_name)[0]
            
            # 补全图片和标注的相对路径
            json_path = os.path.join(data_dir, json_name)
            img_path = None
            for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG']:
                test_path = os.path.join(data_dir, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if not img_path:
                continue

            # 1. 提取数据集原始的真实标注框 (Ground Truth)
            gt_boxes = load_ground_truth_boxes(json_path)
            
            # 2. 驱动 YOLO 进行模型预测 (Prediction)
            yolo_pred = model.predict(source=img_path, imgsz=640, conf=0.25, device='cpu', verbose=False)[0]
            pred_boxes = yolo_pred.boxes.xyxy.cpu().numpy() if yolo_pred.boxes is not None else []

            # 3. 开始使用 Matplotlib 画布拼图 (1行2列的子图对比)
            try:
                img = Image.open(img_path)
            except Exception:
                continue
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # --- 左子图：展示数据集自带的真实手工标注框 (Ground Truth) ---
            ax1.imshow(img)
            ax1.set_title("【真实标注】数据集人工框选区域 (Ground Truth)", fontsize=12, fontweight='bold', color='darkblue')
            ax1.axis('off')
            for box in gt_boxes:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                         linewidth=2, edgecolor='deepskyblue', facecolor='none')
                ax1.add_patch(rect)
                
            # --- 右子图：展示训练好的 YOLO 预测出来的边界框 (Prediction) ---
            ax2.imshow(img)
            ax2.set_title("【模型预测】YOLOv8 深度网络定位 (Prediction)", fontsize=12, fontweight='bold', color='crimson')
            ax2.axis('off')
            for box in pred_boxes:
                xmin, ymin, xmax, ymax = box[:4]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax2.add_patch(rect)

            # 配置整体大标题
            plt.suptitle(f"Task2 目标检测主观评价 — 地标: {landmark} (第 {idx + 1} 组)", fontsize=15, fontweight='bold', y=0.96)
            plt.tight_layout()

            # 保存图像（纯相对路径保存）
            save_filename = f"{landmark}_eval_group_{idx + 1}.png"
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            total_groups += 1
            print(f"成功生成对比图 -> [类别: {landmark:<5} | 组别: {idx+1}/2] -> 保存至: {save_path}")

    print("-" * 70)
    print(f"共计成功生成了 {total_groups} 组对比图。")
    print(f"所有对比图已保存在: {output_dir}/ 目录下")

if __name__ == "__main__":
    run_joint_evaluation()