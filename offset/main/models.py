from torchsummaryX import summary
from net_util import *


# Unet的下采样模块，两次卷积
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, channel_reduce=False):  # 只是定义网络中需要用到的方法
        super(DoubleConv, self).__init__()

        # 通道减少的系数
        coefficient = 2 if channel_reduce else 1

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, coefficient * out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(coefficient * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(coefficient * out_channels, out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


# 上采样（转置卷积加残差链接）
class Up(nn.Module):

    # 千万注意输入，in_channels是要送入二次卷积的channel，out_channels是二次卷积之后的channel
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 先上采样特征图
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels, channel_reduce=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


# simple U-net模型
class U_net(nn.Module):

    def __init__(self):  # 只是定义网络中需要用到的方法
        super(U_net, self).__init__()

        # 下采样
        self.double_conv1 = DoubleConv(3, 32)
        self.double_conv2 = DoubleConv(32, 64)
        self.double_conv3 = DoubleConv(64, 128)
        self.double_conv4 = DoubleConv(128, 256)
        self.double_conv5 = DoubleConv(256, 256)

        # 上采样之前采用回归坐标的方式
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=(1, 1), padding=0)
        self.fc1 = nn.Linear(11264, 128)
        self.fc2 = nn.Linear(128, 8)

        # 上采样
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 16)

        # 最后一层
        # self.conv = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)
        # self.fc1 = nn.Linear(180224, 1024)
        # self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        # down
        # print(x.shape)
        c1 = self.double_conv1(x)  # (,32,512,512)
        p1 = nn.MaxPool2d(2)(c1)  # (,32,256,256)
        c2 = self.double_conv2(p1)  # (,64,256,256)
        p2 = nn.MaxPool2d(2)(c2)  # (,64,128,128)
        c3 = self.double_conv3(p2)  # (,128,128,128)
        p3 = nn.MaxPool2d(2)(c3)  # (,128,64,64)
        c4 = self.double_conv4(p3)  # (,256,64,64)
        p4 = nn.MaxPool2d(2)(c4)  # (,256,32,32)
        c5 = self.double_conv5(p4)  # (,256,32,32)
        # 最后一次卷积不做池化操作

        # up
        # u1 = self.up1(c5, c4)  # (,128,64,64)
        # u2 = self.up2(u1, c3)  # (,64,128,128)
        # u3 = self.up3(u2, c2)  # (,32,256,256)
        # u4 = self.up4(u3, c1)  # (,16,512,512)

        # 最后一层，隐射到3个特征图
        x1 = self.conv1(c5)
        x2 = self.conv2(x1)
        # print(x1.shape)
        x2 = x2.view(x2.size(0), -1)

        # print(x1.shape)
        x = self.fc1(x2)
        out = self.fc2(x)

        return out

    def summary(self, net):
        x = torch.rand(cfg['batch_size'], 3, 352, 512)  # 352*512
        # 送入设备
        x = x.to(cfg['device'])
        # 输出y的shape
        # print(net(x).shape)

        # 展示网络结构
        summary(net, x)


# 主函数调试
if __name__ == "__main__":
    m = U_net().to(cfg['device'])
    m.summary(m)
