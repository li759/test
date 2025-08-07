<sup></sup>[[train_HOPE_sac.input]]
#### 1.1 激光定义
obs['lidar'] 是一个长度为 LIDAR_NUM（如120）的数组。
每个元素代表：以车辆当前位置和朝向为原点，按等角度分布发射的第 $i$条激光射线，在该方向上遇到的最近障碍物的距离（减去车辆自身边界）。
方向分布：从车辆正前方（或某一固定方向）开始，逆时针/顺时针均匀分布一圈。例如，lidar[0] 可能是正前方，lidar[N//4] 是正左方，lidar[N//2] 是正后方，lidar[3N//4] 是正右方。
代码流程：
1. 环境调用：CarParking.render() → CarParking._get_lidar_observation()
2. 仿真计算：LidarSimlator.get_observation() → _rotate_and_filter_obstacles()（障碍物坐标变换）→ _fast_calc_lidar_obs()（距离计算）
3. 输出：obs['lidar']，shape = [LIDAR_NUM]，每个元素为对应方向的距离

#### 1.2 激光方向定义
- 假设 LIDAR_NUM = N，则每个方向的角度为：
    $\large θ_i=\frac {2π}{N}⋅i,i=0,1,...,N−1$
- 每条激光射线的方向为：
    $(cos⁡θ_i,sin⁡θ_i)(cosθ_i​,sinθ_i​)$
#### 1.3 距离计算
- 对每条激光射线 $i$，计算其与所有障碍物的交点，取距离最近的那个，记为 $d_i$​。
- 公式（见 _fast_calc_lidar_obs）：
    $\large d_i=\min\limits_{j}distance(ray_i,obstacle_j))$
- 最终输出为：
    $lidar_i=d_i−vehicle\_boundary_i​$
其中 $vehicle\_boundary_i​$是车辆自身在该方向上的边界距离。
