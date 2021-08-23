import os
import torch

config = {
    # 网络训练部分
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'batch_size': 1,
    'epochs': 510,
    'save_epoch': 100,

    # 设置路径部分
    'train_date': 'all_out',
    'train_way': 'train',
    'test_date': 'all_in',
    'test_way': 'test',

    # 调用的模型
    'pkl_file': 'min_loss_all_in.pth'

}
