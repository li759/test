# 1. 运行主流程（高层）
## 1.1 参数解析与环境初始化

1. 场景选择器初始化
```
def __init__(self) -> None:
	self.scene_types = {0: 'Normal', 1: 'Complex', 2: 'Extrem', 3: 'dlp'}
	self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
	# 记录每种场景的成功率
	self.success_record = {}
	for scene_name in self.scene_types:
		self.success_record[scene_name] = []
```
	功能：管理4种不同难度的场景：Normal、Complex、Extrem、dlp ；根据历史成功率动态选择训练场景； 使用自适应采样策略平衡各场景的训练
2. 环境初始化
	`raw_env = CarParking(fps=100, verbose=verbose)`
	`env = CarParkingWrapper(raw_env)`
	调用链：
	- CarParking.__init__() → 创建基础停车环境
	- CarParkingWrapper.__init__() → 包装环境，添加动作缩放、奖励塑形等功能
3. 智能体初始化
	1. SAC智能体
		```
		configs = {`
		    `discrete': False,` 
		    `'observation_shape': env.observation_shape,` 
		    `'action_dim': env.action_space.shape[0],` 
		    `'hidden_size': 64,` 
		    `'activation': 'tanh',` 
		    `'dist_type': 'gaussian',` 
		    `'actor_layers': actor_params,
		    'critic_layers': critic_params
		    }
		rl_agent = SAC(configs)
		```
		SAC智能体结构：
		- Actor网络：输出动作分布（高斯分布）
		- Critic网络：双Q网络结构，评估状态-动作价值
		- 温度参数：自动调节探索程度
	2.  路径规划器
		```
		step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
		rs_planner = RsPlanner(step_ratio)
		```
		RsPlanner功能：
		- 将Reeds-Shepp路径转换为离散动作序列
		- 管理路径执行状态
	3. Parking智能体
		```parking_agent = ParkingAgent(rl_agent, rs_planner)```
		ParkingAgent功能：
		- 融合RL智能体和路径规划器
		- 在探索阶段使用RL，在接近目标时使用RS路径
	4. 初始化日志、随机种子等
4. 训练循环
	1. 场景选择器（SceneChoose、DlpCaseChoose）
		```
		scene_chosen = scene_chooser.choose_case()
		if scene_chosen == 'dlp':
			case_id = dlp_case_chooser.choose_case()
		else:
			case_id = None
		```
	2. 场景选择策略：
		- 前200个episode：均匀采样
		- 之后：50%概率选择表现最差的场景，50%概率均匀采样
		- 环境重置
		```
		obs = env.reset(case_id, None, scene_chosen)
		parking_agent.reset()
		```
		调用链：
		- env.reset() → CarParking.reset() → ParkingMapNormal.reset()
		- 根据场景类型生成停车环境（普通停车、平行停车、断头路等）
		- 初始化车辆状态和障碍物
	3. 单步执行循环
		```
		while not done:
		    step_num += 1
		    total_step_num += 1
		    探索阶段：随机采样动作
		    if total_step_num <= parking_agent.configs.memory_size and (not parking_agent.executing_rs):
		        action = env.action_space.sample()
				#使用动作掩码确保安全动作
		        action_mask = obs['action_mask']
		        valid_indices = np.where(action_mask > 0.2)[0]
		        # ... 动作验证逻辑
		        log_prob = parking_agent.get_log_prob(obs, action)
		    else:
		        # 训练阶段：使用智能体决策
		        action, log_prob = parking_agent.get_action(obs)
			#环境步进
		    next_obs, reward, done, info = env.step(action)
			#经验回放
		    parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
			#网络更新
		    if total_step_num > parking_agent.configs.memory_size and total_step_num % 10 == 0:
		        actor_loss, critic_loss = parking_agent.update()
		```
	4. 路径规划集成
		if info['path_to_dest'] is not None:
		parking_agent.set_planner_path(info['path_to_dest'])
		RS路径生成：
		- 当车辆距离目标小于RS_MAX_DIST时触发
		- env.find_rs_path() 生成Reeds-Shepp路径
		- parking_agent.set_planner_path() 设置路径给规划器
5. 奖励系统
	1. 奖励组成
	```
		reward_info = OrderedDict({
			'time_cost': reward_list[0], # 时间惩罚
			'rs_dist_reward': reward_list[1], # RS路径距离奖励
			'dist_reward': reward_list[2], # 距离目标奖励
			'angle_reward': reward_list[3], # 角度对齐奖励
			'box_union_reward': reward_list[4] # 重叠面积奖励
		})
	```
	2. 奖励计算
```
		def _get_reward(self, prev_state: State, curr_state: State):
		#计算各种奖励分量
		time_cost = -0.1 # 时间惩罚
		dist_reward = prev_dist - curr_dist # 距离改善奖励
		angle_reward = angle_improvement # 角度改善奖励
		box_union_reward = union_area_improvement # 重叠面积奖励
```
5. 网络更新机制
	1. SAC更新
	```
		def update(self):
			#采样经验批次
			batch = self.memory.sample(self.configs.batch_size)
			#更新Critic网络
			q1_loss = self._update_critic(batch)
			q2_loss = self._update_critic(batch)
			#更新Actor网络
			actor_loss = self._update_actor(batch)
			#更新温度参数
			alpha_loss = self._update_alpha(batch)
			return actor_loss, (q1_loss + q2_loss) / 2
	```
	2. 软更新
	```
		def _soft_update(self, target_net, current_net):
			for target, current in zip(target_net.parameters(), current_net.parameters()):
				target.data.copy_(current.data * self.configs.tau + target.data * (1.0 - self.configs.tau))
	```
6. 评估和保存
	1. 成功率监控
	```
		if info['status'] == Status.ARRIVED:
			succ_record.append(1)
			scene_chooser.update_success_record(1)
		else:
			succ_record.append(0)
			scene_chooser.update_success_record(0)
	```
	 1. 模型保存
		```
		if success_rate_normal >= best_success_rate[0] and ...:	parking_agent.save('%s/SAC_best.pt' % save_path, params_only=True)
		```
		1. 最终评估
		```
		with torch.no_grad():
		    for level in ['dlp', 'Extrem', 'Complex', 'Normal']:
		        env.set_level(level)
      eval(env, parking_agent, episode=eval_episode, log_path=log_path)
		```
7. 关键配置参数
```
	#训练参数
	TRAIN_EPISODE = 100000
	EVAL_EPISODE = 2000
	BATCH_SIZE = 8192
	LR = 5e-06
	TAU = 0.1
	#环境参数
	VALID_SPEED = [-2.5, 2.5]
	VALID_STEER = [-0.62, 0.62]
	RS_MAX_DIST = 10
	TOLERANT_TIME = 200
	#网络参数
	ACTOR_CONFIGS = {
		'n_modal': 2 + int(USE_IMG) + int(USE_ACTION_MASK),
		'lidar_shape': LIDAR_NUM,
		'target_shape': 5,
		'output_size': 2,
		'hidden_size': 256,
		'n_hidden_layers': 3
	}
```

场景选择 → 环境重置 → 观察获取 → 动作决策 → 环境步进 → 奖励计算 → 经验存储 → 网络更新 → 状态检查 → 循环继续
****
## 1.2 关键方法调用
1. 动作决策：
	- parking_agent.get_action(obs)
	- → ParkingAgent.get_action
	- 若未执行规划路径，调用 SACAgent.get_action(obs)
	- 否则用规划器动作
	- → SACAgent.get_action
	- 观测归一化 → tensor → 网络前向 → 分布采样 → mask过滤 → 输出动作、log_prob
2. 环境交互：env.step(action)
	- → CarParkingWrapper.step
	- 动作归一化 → 环境 step → 奖励处理 → 观测处理
3. 经验存储：
	- parking_agent.push_memory(...)
	- → SACAgent.push_memory(...)
4. 智能体更新：
	- parking_agent.update()
	- → SACAgent.update()
	- 采样 batch → 归一化 → 网络前向 → 损失计算 → 反向传播
5. 路径规划：
	- parking_agent.set_planner_path(info['path_to_dest'])
	- → RsPlanner.set_rs_path(...)
****
# 2.输入数据
- obs（dict）：包含 lidar、target、img、action_mask 等
- 动作：连续动作（如转向、速度），或离散动作（mask过滤）
1. 数据流动
	1. 观测归一化：StateNorm.state_norm（src/model/state_norm.py）
	2. 转 tensor：SACAgent.obs2tensor（src/model/agent/sac_agent.py）
	3. 网络前向：MultiObsEmbedding（MLP/Attention，src/model/network.py）
	4. 分布采样：高斯分布采样动作
	5. mask过滤：ActionMask.choose_action（如有，src/model/action_mask.py）
	6. 动作输出：numpy 数组，送入环境
	7. 环境 step：返回 next_obs、reward、done、info
	8. 经验存储：六元组 (obs, action, reward, done, log_prob, next_obs)
	9. 批量采样更新：采样 batch，归一化，网络前向，损失计算，参数更新
2. 输出数据
	- 训练日志：奖励、损失、成功率等
	- 模型权重：定期保存 .pth 文件
- executing_rs = True：表示当前智能体正在执行规划器（如 Reeds-Shepp）生成的路径（即“跟踪规划路径”），而不是用 RL 策略自主决策。
- executing_rs = False：表示当前智能体用 RL 策略自主决策动作。



