[[train_HOPE_sac.input]]
# action_mask 计算流程与公式
1. 生成位置
	- 代码位置：src/model/action_mask.py，类 ActionMask
	- 在环境 CarParking.render 中，调用 self.action_filter.get_steps(observation['lidar']) 得到 action_mask
2. 主要计算步骤
	步骤1：预处理
		对每个离散动作，预先模拟车辆执行该动作若干步后的包络盒（init_vehicle_box、precompute），并计算每个动作每步下各个激光方向的最小安全距离（dist_star）。dist_star[i, j, k] 表示第 i 个激光方向、j 动作、执行 k 步时，车辆与障碍物的最小距离。[1200,42,10]
		对每个激光方向 ii（共 lidar_num 个），生成一条从原点出发、长度为 max_distance 的射线。
			公式：$\large lidar\_line_i=[(0,0),(cos⁡θ_i⋅d,sin⁡θ_i⋅d)]$
			其中 $\large θ_i=\frac{2π}{lidar\_num}⋅i$，d=max_distance
		- 车辆包络盒边的生成
			对每个动作、每步，车辆的包络盒（四边形）都被记录下来。
			每个包络盒有4条边，所有边都被提取出来，shape 变为 (n_action * n_iter * 4, 2, 2)。
		- 计算交点
			对每条激光线和每条车辆包络盒边，计算交点（如果有）。
			交点的距离用欧氏距离公式计算：$d=\sqrt{x^2+y^2}$
			结果 reshape 为 (lidar_num, n_action, n_iter, 4)，每个元素是该激光方向、动作、步数下，4条边的交点距离。
		- 取最大距离
			对每个激光方向、动作、步数，取4条边交点距离的最大值（即车辆包络盒与该激光方向的最远交点）。
			公式：$dist\_star[i,j,k]=\max \limits_{⁡b=1..4}d_{i,j,k,b}$
			np.zeros_like
			np.stack
			np.linalg.norm
			dist_star 是 ActionMask 类中用于加速动作可行性判断的预计算表，它记录了在每个激光方向、每个动作、每个模拟步下，车辆与障碍物的最小距离。
			`vehicle_edges.reshape(-1, 2, 2)`：这行代码的作用是将车辆包络盒的所有边展平成一个二维数组。原始 shape: (42, 10, 4, 2, 2)reshape 后: (1680, 2, 2)，即 42104=1680 条边，每条边2个端点，每个端点2个坐标
			`self.vehicle_boxes = self.init_vehicle_box()`
			`self.dist_star = self.precompute()`
	 步骤2：实时计算
		输入：当前时刻的 lidar 观测（raw_lidar_obs）
		 - 对每个激光方向 $i$，将观测值加上车辆自身边界距离，得到实际可用距离：
			$lidar\_obs_i=clip(raw\_lidar\_obs_i,0,10)+vehicle\_lidar\_base_i$
			`lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base`
		- 距离插值：对 lidar 观测做线性插值，提升分辨率
			`dist_obs = self._linear_interpolate(lidar_obs.reshape(-1), self.up_sample_rate).reshape(-1, 1, 1)`
		- 可行步数判断：对每个动作、每个激光方向，判断车辆执行该动作若干步后是否会碰撞（基于预计算的 dist_star）：
			$step\_save_{i,j,k}=\begin{cases}1, & if \  dist\_star_{i,j,k}\le dist\_obs_i \\0, & otherwise\end{cases}$
			`step_save = np.zeros_like(self.dist_star)`
			`step_save[self.dist_star <= dist_obs] = 1`
			`step_save[self.dist_star > dist_obs] = 0`
		- 最大可行步数：对每个动作，找到所有激光方向上最小的可行步数，归一化后作为 action_mask,
			`max_step = np.argmin(step_save, axis=-1)`
			`max_step[np.sum(step_save, axis=-1) == self.n_iter] = self.n_iter`
			`step_len = np.min(max_step, axis=0)`
			$\large action\_mask_j= \frac{min_{⁡k}(max\_step_{i,j})}{n\_iter}$
			其中 $i$ 为激光方向，$max\_step{i,j}$m​ 为第 $i$ 个方向上动作 $j$ 的最大可行步数。
			`step_save[i, j, :] = [1, 1, 1, 0, 0, 0]：`np.argmin 会返回 3（第一个0出现位置），表示最多能安全执行3步。如果全是1（即始终安全），np.argmin 会返回0，但后面有修正。
			`np.min(max_step, axis=0）`：用于对每个动作在所有激光方向上能安全执行的最大步数取最小值，即每个动作的全局安全步数。
		- 最后经过平滑处理（post_process， 最小滤波），防止 mask 过于激进。
			`step_len = self.post_process(step_len)`
			将 step_len 分为前半部分（forward）和后半部分（backward），对 forward 和 backward 的首尾元素减1，防止边界动作过于激进（即最左/最右、最前/最后的动作更保守）。
			用 minimum_filter1d（scipy 的一维最小值滤波器）对 forward 和 backward 分别做平滑处理，kernel=5。
			用 np.clip 限制在 [0, n_iter] 区间，再除以 n_iter 归一化到 [0, 1]
	步骤3：输出
		输出为 shape = (N_DISCRETE_ACTION,) 的数组，表示每个离散动作的可行性（0~1之间)   

_linear_interpolate：
于原始数组 xx 长度为 NN，上采样倍数为 rr（upsample_rate）：
- 新数组 yy 长度为 N×rN×r
- 对于 yy 的第 jj 个元素（j=0,1,...,N×r−1j=0,1,...,N×r−1）：
    yj=xk⋅(1−α)+xk+1⋅αyj​=xk​⋅(1−α)+xk+1​⋅α
其中
- k=⌊jr⌋k=⌊rj​⌋
- α=jrrα=rjmodr​
- 也就是在 xkxk​ 和 xk+1xk+1​ 之间做线性插值。

