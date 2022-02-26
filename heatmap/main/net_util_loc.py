import torch
import os
import numpy as np
from torch import nn
import torchvision
from config_loc import config_loc as cfg
import torch.utils.data
from torchvision import datasets, transforms, models
import cv2
import PIL
from PIL import Image, ImageFont, ImageDraw
from data_pre_loc import json_to_numpy, generate_heatmaps


# box_3D的数据仓库
class Dataset(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = os.listdir(os.path.join(dataset_path, cfg['train_way'], 'imgs'))

    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_name = self.img_name_list[index]
        img = PIL.Image.open(os.path.join(self.dataset_path, cfg['train_way'], 'imgs', img_name)).convert('RGB')
        img = transforms.ToTensor()(img)  # 3*3000*4096
        img = img[:, :cfg['cut_h'], :cfg['cut_w']]

        img = img.unsqueeze(0)  # 增加一维
        resize = torch.nn.Upsample(scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
        img = resize(img).squeeze(0)  #
        # print(img.shape)

        # 读入标签
        mask_name = img_name.split('.')[0] + '.json'
        mask = json_to_numpy(os.path.join(self.dataset_path, cfg['train_way'], 'labels', mask_name))
        # mask = np.load(os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0] + '.json'),allow_pickle=True)
        # mask = torch.tensor(mask, dtype=torch.float32)

        heatmaps = generate_heatmaps(mask, cfg['input_h'], cfg['input_w'], (cfg['gauss_h'], cfg['gauss_w']))
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return img, heatmaps, img_name

    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)
