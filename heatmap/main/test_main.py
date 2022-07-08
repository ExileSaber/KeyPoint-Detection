from net_util_loc import *
import os
from models_loc import *
import cv2
import PIL
import xml.etree.cElementTree as ET
from xml.etree import ElementTree
import numpy as np
from xml.dom import minidom
import time
# from determine_rotation_angle_loc import calculate_rotation_angle
from determine_location_loc import determine_location
from data_pre_loc import json_to_numpy, generate_heatmaps, heatmap_to_point


def show_point_on_picture(img, landmarks, landmarks_gt):
    for point in landmarks:
        point = tuple([int(point[0]), int(point[1])])
        # print(point)
        img = cv2.circle(img, center=point, radius=20, color=(0, 0, 255), thickness=-1)
    for point in landmarks_gt:
        point = tuple([int(point[0]), int(point[1])])
        # print(point)
        img = cv2.circle(img, center=point, radius=20, color=(0, 255, 0), thickness=-1)
    return img


# 预测
def evaluate(flag=False):
    date = cfg['test_date']
    way = cfg['test_way']

    # 测试路径
    img_path = os.path.join('..', 'data', date, way, 'imgs')
    # 测试集坐标
    label_path = os.path.join('..', 'data', date, way, 'labels')

    # 定义模型
    model = U_net()

    # 034, 128
    model.load_state_dict(torch.load(os.path.join('..', 'weights', cfg['pkl_file'])))
    model.to(cfg['device'])
    # model.summary(model)
    model.eval()

    # 下采样模型
    resize = torch.nn.Upsample(scale_factor=(1, 0.5), mode='bilinear', align_corners=True)

    total_loss = 0
    diff_angle_list = []
    diff_delta_x_list = []
    diff_delta_y_list = []
    max_keypoint_diff = 0

    # 开始预测
    for index, name in enumerate(os.listdir(img_path)):
        print('图像名称：', name+"　　图像编号："+str(index+1))

        # img = cv2.imread(os.path.join(img_path, name))
        # img = cv2.resize(img, (cfg['input_w'], cfg['input_h']))
        # img = transforms.ToTensor()(img)
        # img = torch.unsqueeze(img, dim=0)  # 训练时采用的是DataLoader函数, 会直接增加第一个维度

        img = PIL.Image.open(os.path.join(img_path, name)).convert('RGB')
        img = transforms.ToTensor()(img)  # 3*3000*4096
        img = img[:, :cfg['cut_h'], :cfg['cut_w']]

        img = img.unsqueeze(0)  # 增加一维
        resize = torch.nn.Upsample(scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
        img = resize(img)

        print('输入网络的图片维度信息：', img.shape)

        # 喂入网络
        img = img.to(cfg['device'])

        pre = model(img)
        pre = pre.cpu().detach().numpy()
        # pre = pre.reshape(pre.shape[0], -1, 2)

        pre_point = heatmap_to_point(pre[0])

        point = json_to_numpy(os.path.join(label_path, name.split('.')[0] + '.json'))

        print('图片的jpg文件位置：', os.path.join(img_path, name))
        print('标注的json文件位置：', os.path.join(label_path, name.split('.')[0] + '.json'))

        print('预测的关键点坐标：\n', pre_point)
        print('真实的关键点坐标：\n', point)

        pre_label = torch.Tensor(pre_point.reshape(1, -1)).to(cfg['device'])
        label = torch.Tensor(point.reshape(1, -1)).to(cfg['device'])

        loss_F = torch.nn.MSELoss()
        loss_F.to(cfg['device'])
        loss = loss_F(pre_label, label)  # 计算损失

        print('+++坐标误差损失: ', loss.item())
        total_loss += loss.item()

        if loss.item() > max_keypoint_diff:
            max_keypoint_diff = loss.item()

        print('---------')

        if flag == True:
            del img

            img = cv2.imread(os.path.join(img_path, name))
            img = show_point_on_picture(img, pre_point, point)

            # 存储绘制图像部分
            save_dir = os.path.join('..', 'result', date, way + '_data')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            result_dir = os.path.join(save_dir, name.split('.')[0] + '_keypoint.jpg')
            print('绘制关键点后图像的存储位置：', result_dir)
            cv2.imwrite(result_dir, img)


    print('##################')
    print('# ---- Mean ---- #')
    print('##################')

    print('平均每个关键点坐标误差：', total_loss / (index + 1), '　　最大单个关键点坐标误差：', max_keypoint_diff)


if __name__ == "__main__":
    # choose weights
    # choose_weights()

    # 对一组权重进行预测
    evaluate(flag=True)
