# Industry-Keypoint-Detection
基于自己标注的工业图像的关键点检测，每张图片标注了4个关键点，采用的U-net网络

## 主要内容
* 这部分主要是个人第一次做目标检测方面的任务，用于练手和理解网络
* 网络采用的是U-net
* 标签构建采用的Coordinate方法，损失函数仅采用了坐标点之间的距离平方和

## 之后的探索过程
* 标签构建尝试Heatmap和Heatmap + Offsets
* 网络结构改进
