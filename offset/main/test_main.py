from net_util import *
import os
from models import *
import cv2
import xml.etree.cElementTree as ET
from xml.etree import ElementTree
import numpy as np
from xml.dom import minidom
import time
from determine_rotation_angle import calculate_rotation_angle
from data_pre import one_json_to_numpy


def show_point_on_picture(img, landmarks, landmarks_gt):
    for point in landmarks[0]:
        point = tuple([int(point[0]), int(point[1])])
        # print(point)
        img = cv2.circle(img, center=point, radius=20, color=(0, 0, 255), thickness=-1)
    for point in landmarks_gt:
        point = tuple([int(point[0]), int(point[1])])
        # print(point)
        img = cv2.circle(img, center=point, radius=20, color=(0, 255, 0), thickness=-1)
    return img


# 预测
def evaluate():
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
    # resize = torch.nn.Upsample(scale_factor=(1, 0.5), mode='bilinear', align_corners=True)

    diff_angle_list = []
    total_loss = 0
    # 开始预测
    for index, name in enumerate(os.listdir(img_path)):
        print(name+": "+str(index+1))

        img = cv2.imread(os.path.join(img_path, name))
        img = cv2.resize(img, (512, 352))
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img, dim=0)  # 训练时采用的是DataLoader函数, 会直接增加第一个维度
        print(img.shape)

        # 喂入网络
        img = img.to(cfg['device'])

        pre = model(img)
        pre = pre.cpu().detach().numpy()
        pre = pre.reshape(pre.shape[0], -1, 2)

        gt_point = one_json_to_numpy(os.path.join(label_path, name.split('.')[0] + '.json'))
        gt_point = gt_point.reshape(-1, 2)

        print(os.path.join(img_path, name))
        print(os.path.join(label_path, name.split('.')[0] + '.json'))

        print(pre)
        print(gt_point)

        pre_label = torch.Tensor(pre.reshape(1, -1)).to(cfg['device'])
        label = torch.Tensor(gt_point.reshape(1, -1)).to(cfg['device'])

        loss_F = torch.nn.MSELoss()
        loss_F.to(cfg['device'])
        loss = loss_F(pre_label, label)  # 计算损失

        print('坐标误差损失: ', loss.item())
        total_loss += loss.item()
        print('---------')

    #     del img
    #
    #     img = cv2.imread(os.path.join(img_path, name))
    #     print(img.shape)
    #     img = show_point_on_picture(img, pre, gt_point)
    #
    #     # 存储绘制图像部分
    #     save_dir = os.path.join('..', 'result', date, way + '_data')
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     result_dir = os.path.join(save_dir, name.split('.')[0] + '_keypoint.jpg')
    #     print(result_dir)
    #     cv2.imwrite(result_dir, img)
    #
    #     # 通过预测的关键点计算旋转角度
    #     keypoints = pre[0].tolist()
    #     pre_angle, pre_turn = calculate_rotation_angle(keypoints)
    #     print(pre_angle, pre_turn, sep='  ')
    #
    #     # 通过真实的关键点计算旋转角度
    #     keypoints = gt_point.tolist()
    #     true_angle, true_turn = calculate_rotation_angle(keypoints)
    #     print(true_angle, true_turn, sep='  ')
    #
    #     if pre_angle < 0:
    #         diff_angle = -100000
    #     else:
    #         diff_angle = abs(true_angle - pre_angle)
    #     print(diff_angle)
    #
    #     diff_angle_list.append(diff_angle)
    #     print('=========')
    #
    # print(np.mean(diff_angle_list))

    print(total_loss / (index + 1))


if __name__ == "__main__":
    # choose weights
    # choose_weights()

    # 对一组权重进行预测
    evaluate()
