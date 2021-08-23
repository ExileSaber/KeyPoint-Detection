import os
import torch

config = {
    # 网络训练部分
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'batch_size': 1,
    'epochs': 510,
    'save_epoch': 100,
    'learning_rate': 0.0001,
    'lr_scheduler': 'step1',  # 可以选择'step1','step2'梯度下降，'exponential'指数下降

    # 原图尺寸
    'img_h': 3000,
    'img_w': 4096,

    # 裁剪后的尺寸
    'cut_h': 2944,
    'cut_w': 4096,

    # 网络输入的图像尺寸
    'input_h': 736,
    'input_w': 1024,

    # 网络评估部分
    'test_batch_size': 1,
    'test_threshold': 0.5,

    # 设置路径部分
    'train_date': '08_11_in',
    'train_way': 'train',
    'test_date': '08_11_in',
    'test_way': 'test',

    # 调用的模型
    'pkl_file': '0820_min_loss_all_in_heatmap.pth'

}
