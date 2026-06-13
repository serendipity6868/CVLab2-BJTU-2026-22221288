# -*- coding: gbk -*-
import os
# 阻止全局 Python 级别警告
os.environ["PYTHONWARNINGS"] = "ignore"  

import sys
import io
import warnings
import logging
import torch
import torchvision

# 忽略 Python 的标准告警提示
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=UserWarning)

# 保持原有策略：防止 Windows 终端下的 GBK 打印报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 允许 ultralytics 基础日志流出（保留表头和指标）
logging.getLogger("ultralytics").setLevel(logging.INFO)
logging.getLogger("ultralytics.data.utils").setLevel(logging.ERROR)
logging.getLogger("ultralytics.utils.checks").setLevel(logging.ERROR)

# ==================== ?? 核心修复：纯 Python 实现的 NMS 替换补丁 ====================
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

# 【核心置换】用自己写的免编译 Python NMS 替换掉官方那个会崩溃的 C++ NMS
torchvision.ops.nms = python_nms
# ===================================================================================

from ultralytics import YOLO
import yaml

def train_yolo_model():
    # 读取你原本干净的 dataset.yaml
    yaml_relative_path = "Task2/dataset.yaml"
    with open(yaml_relative_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # 在当前 Code 目录下动态创建一个不含任何中文注释的临时纯净 yaml 文件
    temp_yaml_path = "Task2/temp_dataset.yaml"
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False)
        
    # 初始化 YOLOv8 模型
    model = YOLO("yolov8n.pt")
    
    print("====== 成功加载动态兼容补丁 ======")
    print("====== 启动 YOLO 文字检测模型训练 ======")
    
    try:
        # 启动训练
        results = model.train(
            data=temp_yaml_path,           
            epochs=50,                    
            imgsz=640,                    
            batch=16,                     
            workers=2,                    
            project="Task2/runs",         
            name="text_detection",        
            device='cpu',                 
            patience=5,
            verbose=True                  
        )
    finally:
        # 训练结束或报错退出时，自动清理掉这个临时生成的辅助文件
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
    
    print("\n训练与评估结束！")
    print("最佳权重保存在: Task2/runs/text_detection/weights/best.pt")

if __name__ == "__main__":
    train_yolo_model()