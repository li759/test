[[train_HOPE_sac.input]]
# 1. img数据的获取与生成流程
1. 主要文件和函数
	- 文件：src/env/car_parking_base.py
	- 函数：CarParking.render()
	- 图像获取函数：CarParking._get_img_observation(surface)
	- 作用：以车辆为中心旋转、裁剪环境图像，得到车辆视角的RGB图像。
	- 图像预处理函数：CarParking._process_img_observation(img)
	- 作用：下采样、归一化、通道处
	- 图像处理类：Obs_Processor（src/env/observation_processor.py）
# 2. 计算公式与处理步骤
1. 车辆视角图像生成
	- 以车辆为中心旋转环境图像，使车辆始终朝上（正方向），并裁剪出以车辆为中心的窗口。
	- 得到 shape 为 (OBS_W, OBS_H, 3) 的RGB图像。
 2. 图像预处理
	- 背景色处理：将背景色像素设为黑色。
	- 下采样：用 cv2.resize 将图像缩小（如原始256×256变为64×64）。
	- 归一化：像素值除以255，归一化到[0,1]。
	- 通道顺序：通常为 (H, W, C)，后续网络可能会转为 (C, H, W)。
	公式： $\large img_{norm}=\frac{cv2.resize(change\_bg\_color(img_{raw}))}{255.0}$
# 3. 每一维的物理意义
	- img 是车辆视角下的环境RGB图像，反映了车辆周围的空间布局、障碍物、目标车位等。
	- 每个像素：表示车辆视角下某个空间位置的颜色信息。
	- 三个通道：分别为R、G、B。
# 4. 数据流动总结
1. 环境 reset/step/render 时，调用 CarParking.render()。
2. render() 调用 get_img_observation() 获取车辆视角RGB图像。
3. process_img_observation() 调用 Obs_Processor.process_img() 进行下采样、归一化等处理。
4. 处理后的图像作为 obs['img']，供后续神经网络输入。