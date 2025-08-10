[[train_HOPE_sac]]

| 算法库         | 主要功能   | 数学公式       | 源码位置                |
| ----------- | ------ | :--------- | :------------------ |
| PyTorch     | 深度学习框架 | 自动微分、张量运算  | 全项目                 |
| SAC         | 强化学习算法 | 熵正则化、双Q网络  | sac_agent.py        |
| Transformer | 注意力机制  | 自注意力、多头注意力 | attention.py        |
| Reeds-Shepp | 路径规划   | 圆弧-直线组合    | reeds_shepp.py      |
| Shapely     | 几何计算   | 多边形相交、面积计算 | car_parking_base.py |
| NumPy       | 数值计算   | 矩阵运算、随机采样  | 全项目                 |
| Matplotlib  | 可视化    | 绘图、动画      | 训练脚本                |

*****
1. Pytorch
2. SAC
	- SAC算法核心流程与数学公式
		1. 算法核心思想
			最大化奖励的同时最大化策略熵（鼓励探索）。
			双Q网络防止Q值高估，目标网络提升稳定性。
			自动温度调整，平衡探索与利用。
		2. 核心数学
			- 策略目标
				最大化：$J(π)=E_{(s_{t},a_{t})}∼ρ_{π}[r(s_{t},a_{t})+αH(π(⋅∣s_{t}))])$
				其中$H(π(⋅∣s_{t}))=−E_{a_{t}}∼πlog⁡π(a_{t}∣s_{t})$为熵。
			- Q函数目标
				$J_{Q}(θ)=E_{(s_{t},a_{t})∼D}[ \frac{1}{2}(Q_{θ}(s_{t},a_{t})−(r_{t}+γE_{a_{t+1}∼π}[Q_{θˉ}(s_{t+1},a_{t+1})−αlog⁡π(a_{t+1}∣s_{t+1})]))^2]$
			- 策略网络目标
				$J_{π​}(ϕ)=E_{s_t​∼D,a_{t}​∼π_{ϕ}​​}[αlogπ_{ϕ}(a_{t}​∣s_{t}​)−Q_{θ​}(s_{t}​,a_{t}​)]$
			- 温度参数目标
				$J(α)=E_{a_{t}​∼π​}[−αlogπ(a_{t}​∣s_{t}​)−αH_{target}]​$
	- 源码实现流程与方法说明
		1. 主要库
			torch：神经网络与自动微分
			torch.optim：优化器
			torch.distributions.Normal：高斯策略分布
			numpy：数值计算
		2. SACAgent类（src/model/agent/sac_agent.py）
			- 初始化网络与优化器
				`self.actor_net = MultiObsEmbedding(self.configs.actor_layers).to(self.device)`
				`self.critic_net1 = SACCriticAdapter(self.configs.critic_layers).to(self.device)`
				`self.critic_net2 = SACCriticAdapter(self.configs.critic_layers).to(self.device)`
				`self.critic_target_net1 = deepcopy(self.critic_net1)`
				`self.critic_target_net2 = deepcopy(self.critic_net2)`
				`self.actor_optimizer = torch.optim.Adam([...])`
				`self.critic_optimizer1 = torch.optim.Adam(self.critic_net1.parameters(), ...)`
				`self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), ...)`
				`self.log_alpha = torch.tensor(np.log(self.configs.initial_temperature)).to(self.device)`
				`self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], ...)`
			- 采样动作
				`def get_action(self, obs):`
					`dist = self._actor_forward(obs)`
					`action, log_prob = self._post_process_action(dist)`
					`return (action, log_prob)`
			- 经验存储
				`def push_memory(self, observations):`
					`self.memory.push(observations)`
			- 参数更新
				`def update(self):`
					`采样批次`
					`batches = self.memory.sample(self.configs.batch_size)`
					`计算目标Q值`
					`q_target = reward_batch + (1 - done_batch) * gamma * (min(q1_target, q2_target) - alpha * next_log_prob)`
					`Critic损失`
					`q1_loss = F.mse_loss(current_q1, q_target.detach())`
					`q2_loss = F.mse_loss(current_q2, q_target.detach())`
					`Actor损失`
					`actor_loss = (alpha * log_prob - min(q1_value, q2_value)).mean()`
					`温度损失`
					`alpha_loss = (alpha * (-log_prob - target_entropy).detach()).mean()`
					`软更新目标网络`
					`self.soft_update(self.critic_target_net1, self.critic_net1)`
					`self._soft_update(self.critic_target_net2, self.critic_net2)`
					`return (actor_loss, q1_loss.item())`
	- 网络结构（src/model/network.py）
		策略网络/价值网络：MultiObsEmbedding，支持多模态输入（激光、图像、目标、动作掩码等），可选注意力机制。
3. SAC训练主流程（伪代码+源码结合）
	`for each training step:`
		`# 1. 采样动作`
		`action, log_prob = agent.get_action(obs)`
		`# 2. 环境交互`
		`next_obs, reward, done, info = env.step(action)`
		`# 3. 存储经验`
		`agent.push_memory((obs, action, reward, done, log_prob, next_obs))`
		`# 4. 批量采样经验，更新网络`
		`if enough samples:`
		`actor_loss, critic_loss = agent.update()`
 4. 源码与公式一一对应关系

| 数学公式      | 源码方法                                                                  | 说明        |
| --------- | --------------------------------------------------------------------- | --------- |
| $J_Q​$    | `q1_loss = F.mse_loss(current_q1, q_target.detach())`                 | Q网络均方误差损失 |
| $J_π$​    | `actor_loss = (alpha * log_prob - min(q1_value, q2_value)).mean()`    | 策略网络损失    |
| $J_{(α)}$ | `alpha_loss = (alpha * (-log_prob - target_entropy).detach()).mean()` | 温度参数损失    |
| 软更新       | `self._soft_update(self.critic_target_net1, self.critic_net1)`        | 目标网络软更新   |
****

SAC（Soft Actor-Critic）智能体的核心结构可以概括为“一个Actor（策略网络）+两个Critic（Q网络）+一套熵正则化机制”，整体遵循Actor-Critic范式，但在目标函数、网络设计和更新方式上引入了最大熵强化学习的思想，显著提升了样本效率、训练稳定性及探索能力。
1. Actor（策略网络）
	- 输入：当前状态 $s_t$
	- 输出：连续动作 $a_t$ （用于和环境交互）及其（一个可导的随机策略）对数概率 $log π_φ(a_t|s_t)$
	- 网络结构：  
		共享主干：2层全连接（如100-100）+$ReLU$
		双头输出：均值头 $μ(s_t)$ ；对数标准差头 $log σ(s_t)$（通常裁剪到[-20,2]）
	- 采样：利用重参数化技巧（rsample）从 N(μ,σ) 采样后经 tanh 压缩到 [-1,1]，同时计算可导的 log-prob 用于熵正则化
2. Critic（双Q网络）
- 两套结构完全相同的Q网络：$Q_{θ1}$、$Q_{θ2}$，用于缓解过高估计。
	- 输入：状态-动作拼接 [s_t, a_t]
	- 网络结构：2层全连接（如256-256）+ReLU，输出单一标量 Q 值
    - 目标Q计算：  Q_target = r + γ ( min(Q_θ1′,Q_θ2′) – α log π_φ )  
	    其中 Q′ 表示缓慢更新的目标网络，α 是自适应温度系数
3. 温度系数 α 的自适应机制
	- 引入可训练参数 log α，通过最小化以下损失自动调节：  
		loss_α = – log α * (log π_φ + target_entropy).detach()  
		其中 target_entropy 一般设为 ‑|A|（动作维度负数）以保证策略充分随机
4. 经验回放与训练流程
	- 交互数据存入Replay Buffer。
每次随机采样一批数据，按以下顺序更新：  
a. 数据流  ：每次从经验回放随机采样一个 $batch (s, a, r, s^′, d)$。
b.先更新双Critic:用 target 网络计算 y：  $y = r + γ (1 – d) [ min(Q_{θ}1^′, Q_{θ}2^′) – α log π_φ(a^′|s^′) ]$  Critic 的 MSE 损失：  $L_Q = Σ (Q_{θi}(s,a) – y)² , i=1,2$
c. Actor损失: $L_π = E[ α log π_φ(a\_new|s) – min(Q_{θ1}(s,a\_new), Q_{θ2}(s,a\_new)) ]$
d. 最后更新温度系数α损失$:L_α = –E[ α log π_φ(a|s) + α · target\_entropy$ ]；  
e. 软更新目标网络:θ′ ← τ θ + (1–τ) θ′ ，τ≈0.005

****

1. 观测归一化
	$observation_{norm}​=\frac{observation−μ​}{σ+ϵ}$
	$μ$:​：当前模态的均值（self.state_mean[obs_type]）
	$σ$：当前模态的标准差（self.state_std[obs_type]）
	$ϵ$：极小常数，防止除零（源码中为 1e−81e−8）
	$μ_{new}​=μ_{old}​+\frac{x−μ_{old}}{n}​​$
	$S_{new}=S_{old}+(x−μ_{old})⋅(x−μ_{new})$
	$σ=\sqrt{\frac{S}{n}}​​$
2. 转 tensor：
	- 单样本（dict）：
		 obs[obs_type]→torch.FloatTensor(obs[obs_type])→unsqueeze(0)
		 即 shape 从 (feature_dim,) 变为 (1, feature_dim)
	- 批量（list）：
		 对每个模态，收集所有样本，堆叠成 numpy 数组，再转为 tensor merged_obs[obs_type]=torch.FloatTensor(np.array([o[obs_type] for o in obs]))
		 shape 从 (batch, feature_dim)
3. 网络前向：
	- 各模态特征嵌入（Embedding）
		每种模态通过独立的线性层（或卷积层）映射到统一的嵌入维度 embed_size，如128。
		以 lidar 为例：$\large f_{lidar}=ϕ(x_{lidar}⋅W_{lidar,1}+b_{lidar,1})⋅W_{lidar,2}+b{lidar,2}$
		其中 $ϕ$ 是激活函数（如 Tanh/LeakyReLU）
	- 特征融合
		1. 方式一：MLP（拼接）
			将所有模态的嵌入特征拼接在一起，形成一个长向量。
			输入 shape: (batch, n_modal × embed_size)
			送入 MLP 进一步提取高阶特征。
			$f_{all}​=Concat(f_{lidar​},f_{target}​,…)$
			$h_1=ϕ(f_{all}⋅W_1+b_1)$
			$h_2=ϕ(h_1⋅W_2+b_2)$
			$output=h_2⋅W_3+b_3$
		2. 方式二：Attention（堆叠）
			将各模态特征堆叠成 (batch, n_modal, embed_size)
			输入送入 AttentionNetwork（即 Transformer 编码器）
			Attention 机制自动建模模态间的相关性
			Attention 公式（以单层为例）：
			- QKV变换：$[Q,K,V]=X⋅W_{qkv}​$
				其中 $X \ shape = (batch, n\_modal, embed\_size)$
				self.to_qkv 是一个线性层，将输入映射到 3 × (heads × dim_head) 维度
				然后 chunk(3, dim=-1) 分为 Q、K、V
			- 多头拆分
				$\large Q,K,V∈R^{batch×heads×n×dim_head}$
			- 注意力分数计算
				$scores=\frac{K_T}{\sqrt{dim_{head}}}$
				`dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale`
			- softmax归一化
				$α=softmax(scores,dim=−1)$
				`attn = self.attend(dots)`
			- dropout（训练时生效）
				`attn = self.dropout(attn)`
			- 加权求和
				$O=αV$
				`out = torch.matmul(attn, v)`
			- 多头合并
				$O_{flat}=reshape(O,[batch,n,heads×dim_head])$
				`out = rearrange(out, 'b h n d -> b n (h d)')`
			- 输出线性层
				$output=O_{flat}⋅W_{out}+b_{out}​$
****
