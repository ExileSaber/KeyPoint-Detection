from models_loc import *
from config_loc import config_loc as cfg


# 训练主函数入口
def train():
    print('start')

    save_epoch = cfg['save_epoch']
    min_avg_loss = 5000
    date = cfg['train_date']

    # 模型
    model = U_net()

    # 读入权重
    start_epoch = 0

    # 加载初始模型
    if cfg['use_old_pkl'] is True:
        model.load_state_dict(torch.load(os.path.join('..', 'weights', cfg['old_pkl'])))
        print('模型加载完成')

    model.to(cfg['device'])
    model.summary(model)

    start_epoch += 1

    # 数据仓库`
    dataset = Dataset(os.path.join('..', 'data', cfg['train_date']))

    train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=cfg['batch_size'],
                                                    shuffle=True)

    # 优化器
    loss_F = torch.nn.MSELoss()
    loss_F.to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    # 添加学习率衰减

    max_loss_name = ''
    for epoch in range(start_epoch, cfg['epochs'], 1):
        # model.train()
        total_loss = 0.0
        min_loss = 1000
        max_loss = 0.001
        # 按批次取文件
        for index, (x, y, img_name) in enumerate(train_data_loader):
            img = x.to(cfg['device'])
            label = y.to(cfg['device'])

            # print('------------')
            # print(img.shape)
            # print('------------')

            pre = model(img)  # 前向传播
            # 计算损失反向传播
            # print(pre.shape)
            # print('------------')
            # print(label.shape)

            loss = loss_F(pre, label)  # 计算损失
            optimizer.zero_grad()  # 因为每次反向传播的时候，变量里面的梯度都要清零
            loss.backward()  # 变量得到了grad
            optimizer.step()  # 更新参数
            total_loss += loss.item()

            if loss < min_loss:
                min_loss = loss

            if loss > max_loss:
                max_loss = loss
                max_loss_name = img_name[0]

            # if (index+1) % 5 == 0:
            #     print('Epoch %d loss %f' % (epoch, total_loss / (index + 1)))

        avg_loss = total_loss/(index+1)

        if avg_loss < min_avg_loss:
            min_avg_loss = avg_loss
            torch.save(model.state_dict(), os.path.join('..', "weights", 'min_loss.pth'))
            # if cfg['pytorch_version'] is False:
            #     torch.save(model.state_dict(), os.path.join('..', "weights", 'old_version_min_loss.pth'), _use_new_zipfile_serialization=cfg['pytorch_version'])

        print('Epoch %d, photo number %d,avg loss %f, min loss %f, max loss %f, max loss name %s,min avg loss %f' % (epoch, index+1, avg_loss, min_loss, max_loss, max_loss_name, min_avg_loss))

        print('-------------------')

        # 跑完save_epoch个epoch保存权重
        # save_name = "epoch_" + str(epoch).zfill(3) + ".pth"
        # if (epoch) % save_epoch == 0:
        #     torch.save(model.state_dict(), os.path.join('..', "weights", save_name))


if __name__ == "__main__":
    # 训练
    train()
