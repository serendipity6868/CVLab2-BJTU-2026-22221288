# -*- coding: gbk -*-
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Extractor(nn.Module):
    def __init__(self):
        super(ResNet50Extractor, self).__init__()
        # 加载预训练的 ResNet-50 模型
        resnet = models.resnet50(pretrained=True)
        
        # 移除最后的全连接层 (fc)，保留到自适应平均池化层 (avgpool)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # 输入形状: [batch_size, 3, 224, 224]
        features = self.feature_extractor(x)
        # 此时形状为 [batch_size, 2048, 1, 1]，展平为 [batch_size, 2048]
        features = torch.flatten(features, 1)
        
        # L2 归一化：将特征向量单位化，后面计算矩阵乘法直接就是余弦相似度
        features = nn.functional.normalize(features, p=2, dim=1)
        return features