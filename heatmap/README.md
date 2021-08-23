## KeyPoint-Detection/heatmap
采用高斯图作为网络的输出结果，模型预测的关键点精度进一步提高，误差控制在几个像素
在data文件夹中给出了一张实例图片和标注的json文件

## 主要内容
* 使用了高斯图作为label，网络的输出结果为对同等维度的高斯图
* 网络模型采用的是 U-net
* 损失函数使用的是 torch.nn.MSELoss()
* 在网络预测的高斯图中采用最大值点作为预测的关键点位置

## 模型效果
其中绿色点为标注的关键点，红色点为预测的关键点
<img src="https://github.com/ExileSaber/KeyPoint-Detection/blob/main/heatmap/result/08_11/test/202108110759_6_keypoint.jpg" width="800" height="600" alt="模型效果"/><br/>
