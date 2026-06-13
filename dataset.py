# -*- coding: gbk -*-
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageRetrievalDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: base 文件夹或 query 文件夹的路径
        """
        self.data_dir = data_dir
        self.img_names = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 使用标准的 ImageNet 预处理，以契合预训练 ResNet-50
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 根据文件名解析出类别标签 (例如 "fhy-xxx.jpg" -> "fhy")
        # 训练/提取时不用这个标签，仅在评测匹配时使用
        label = img_name.split('-')[0]
        
        return image, img_name, label