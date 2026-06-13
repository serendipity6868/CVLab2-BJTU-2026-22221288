# -*- coding: gbk -*-
import matplotlib
import matplotlib.pyplot as plt
import torch
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def evaluate_and_plot_landmarks(query_data, base_data, max_k=60):
    query_features = query_data['features']  # [N_query, 2048]
    query_labels = query_data['labels']      # 长度为 N_query 的列表
    
    base_features = base_data['features']    # [N_base, 2048]
    base_labels = base_data['labels']        # 长度为 N_base 的列表
    
    # 1. 计算余弦相似度矩阵 [N_query, N_base]
    similarity_matrix = torch.mm(query_features, base_features.t())
    
    # 2. 对每个 query 获取前 max_k 个最高相似度的 base 索引
    _, topk_indices = torch.topk(similarity_matrix, k=max_k, dim=1, largest=True, sorted=True)
    
    # 3. 获取所有唯一的 Landmark 类别名称（共 12 个）
    landmarks = sorted(list(set(query_labels)))
    print(f"检测到共有 {len(landmarks)} 个 Landmark 类别: {landmarks}\n")
    
    # 定义 1-60 的 K 值列表
    k_values = list(range(1, max_k + 1))
    
    # 创建一个大画布，画 3行4列 共 12 个子图
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten() # 展平一维方便循环
    
    # 4. 分类别计算 P@K 曲线
    for idx, landmark in enumerate(landmarks):
        # 找出当前属于该 landmark 的所有 query 索引
        q_indices = [i for i, label in enumerate(query_labels) if label == landmark]
        num_q = len(q_indices)
        
        # 存储当前类在不同 K 下的平均准确率
        landmark_p_at_k = []
        
        for k in k_values:
            total_precision = 0.0
            for q_idx in q_indices:
                correct_count = 0
                # 检查当前 query 的前 K 个检索结果
                for b_idx_tensor in topk_indices[q_idx][:k]:
                    b_label = base_labels[b_idx_tensor.item()]
                    if b_label == landmark:
                        correct_count += 1
                total_precision += (correct_count / k)
            
            # 计算当前 K 值下该类别的平均 Precision
            avg_precision = (total_precision / num_q) * 100
            landmark_p_at_k.append(avg_precision)
            
        # 5. 开始绘制当前地标的子图
        ax = axes[idx]
        ax.plot(k_values, landmark_p_at_k, color='dodgerblue', linewidth=2, label=f'{landmark}')
        ax.set_title(f'地标: {landmark} (Query数:{num_q})', fontsize=12, fontweight='bold')
        ax.set_xlabel('K 值', fontsize=10)
        ax.set_ylabel('Precision (%)', fontsize=10)
        ax.set_xlim(1, max_k)
        ax.set_ylim(0, 105) # 百分比范围 0-100%
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 特定的 TopK 关键点
        print(f"[{landmark}] P@20: {landmark_p_at_k[19]:.2f}% | P@40: {landmark_p_at_k[39]:.2f}% | P@60: {landmark_p_at_k[59]:.2f}%")

    # 整体美化并展示
    plt.suptitle('各 Landmark 类别图片检索 Precision@K 趋势曲线图', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 自动保存大图到 Task1 文件夹
    os.makedirs("Task1", exist_ok=True)
    plt.savefig("Task1/landmarks_p_at_k_curves.png", dpi=300)
    print("\n[成功] 12类地标折线图已合并保存至: Task1/landmarks_p_at_k_curves.png")
    plt.show()

if __name__ == "__main__":
    # 加载前面提取好的特征
    try:
        query_data = torch.load("Task1/query_features.pth")
        base_data = torch.load("Task1/base_features.pth")
        
        # 执行 12 类别曲线评测与画图
        evaluate_and_plot_landmarks(query_data, base_data, max_k=60)
        
    except FileNotFoundError:
        print("未找到特征文件！请先在 Code 目录下运行 `python Task1/extract_features.py` 提取特征。")