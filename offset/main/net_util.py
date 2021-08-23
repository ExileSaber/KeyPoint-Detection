import torch
import os
import numpy as np
from torch import nn
import torchvision
from config import config as cfg
import torch.utils.data
from torchvision import datasets, transforms, models
import cv2
from data_pre import one_json_to_numpy


# box_3D的数据仓库
class Dataset(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = os.listdir(os.path.join(dataset_path, 'imgs'))

    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 352))
        img = transforms.ToTensor()(img)

        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        mask = one_json_to_numpy(label_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        mask = torch.tensor(mask, dtype=torch.float32)

        # print(img_path)
        # print(label_path)
        # print('-----------------')
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")

        return img, mask

    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)
