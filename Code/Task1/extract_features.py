# -*- coding: gbk -*-
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from dataset import ImageRetrievalDataset
from model import ResNet50Extractor
import os
import time  # 引入时间模块用于性能统计

def extract_and_save(data_dir, save_path, device):
    dataset = ImageRetrievalDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    model = ResNet50Extractor().to(device)
    model.eval()
    
    all_features = []
    all_names = []
    all_labels = []
    
    # 用于可视化的统计数据列表
    batch_indices = []
    batch_times = []
    accumulated_samples = []
    total_samples = 0
    
    print(f"\n开始提取图片特征: {data_dir}")
    print("-" * 50)
    
    start_time = time.time()
    
    with torch.no_grad():
        # enumerate 可以带出当前的迭代步数 (batch_idx)
        for batch_idx, (images, names, labels) in enumerate(dataloader):
            batch_start = time.time()
            
            images = images.to(device)
            features = model(images) # 提取归一化特征
            
            all_features.append(features.cpu())
            all_names.extend(names)
            all_labels.extend(labels)
            
            # 记录当前批次的耗时与样本数
            batch_end = time.time()
            elapsed = batch_end - batch_start
            total_samples += len(names)
            
            batch_indices.append(batch_idx + 1)
            batch_times.append(elapsed)
            accumulated_samples.append(total_samples)
            
            # 在终端打印当前迭代轮数/步数进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"迭代进度: Batch [{batch_idx + 1}/{len(dataloader)}] | "
                      f"当前批次耗时: {elapsed:.3f}s | "
                      f"累计已提取图片: {total_samples}/{len(dataset)}")
                
    # 拼接并保存
    all_features = torch.cat(all_features, dim=0)
    data_to_save = {
        'features': all_features,
        'names': all_names,
        'labels': all_labels
    }
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data_to_save, save_path)
    
    print("-" * 50)
    print(f"提取完成！总耗时: {time.time() - start_time:.2f}s")
    print(f"特征已成功保存至 {save_path}, 形状: {all_features.shape}")
    
    # ==================== 开始绘制迭代过程可视化图 ====================
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 绘制左轴：每个 Batch 的提取耗时
    color = 'tab:blue'
    ax1.set_xlabel('迭代轮数 (Batch 数量)', fontsize=12)
    ax1.set_ylabel('每批次耗时 (秒)', color=color, fontsize=12)
    ax1.plot(batch_indices, batch_times, color=color, marker='o', linestyle='-', linewidth=1.5, label='单批次耗时')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制右轴：累计提取的图片数量趋势
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('累计提取图片总数 (张)', color=color, fontsize=12)
    ax2.plot(batch_indices, accumulated_samples, color=color, linestyle='--', linewidth=2, label='累计图片数')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 加上标题
    dir_name = os.path.basename(os.path.normpath(data_dir))
    plt.title(f'数据集 [{dir_name}] 特征提取迭代过程可视化', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    # 保存可视化图片到本地并展示
    img_save_name = f"Task1/{dir_name}_extraction_progress.png"
    plt.savefig(img_save_name, dpi=300)
    print(f"迭代统计图已保存至: {img_save_name}")
    plt.show() # 会自动弹窗显示图表
    # ==================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 在 Code 目录下运行，保持相对路径一致
    base_dir = "BJTU-DataSet-2026/image_retrieval/base/BJTU"
    query_dir = "BJTU-DataSet-2026/image_retrieval/query"
    
    # 提取并保存特征（每次提取完会自动弹出可视化图表，关闭当前图表后才会继续下一个）
    extract_and_save(base_dir, "Task1/base_features.pth", device)
    extract_and_save(query_dir, "Task1/query_features.pth", device)