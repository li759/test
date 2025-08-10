[[train_HOPE_sac]]
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

SAC（Soft Actor-Critic）智能体的核心结构可以概括为“一个Actor（策略网络）+两个Critic（Q网络）+一套熵正则化机制”，整体遵循Actor-Critic范式，但在目标函数、网络设计和更新方式上引入了最大熵强化学习的思想，显著提升了样本效率、训练稳定性及探索能力。
1. Actor（策略网络）
	- 输入：当前状态 $s_t$
	- 输出：连续动作 $a_t$ （用于和环境交互）及其对数概率 $log π_φ(a_t|s_t)$
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
a. 先更新双Critic（最小化Bellman MSE）；  
b. 再更新Actor（最大化 Q – α log π）；  
c. 最后更新温度系数α；  
d. 软更新目标网络
****
- Actor（策略网络）
	1. 前向运算 + 公式推导：
		Step 1 主干网络P：$h = relu(W_{2}  relu(W_{1}  s_t + b_{1}) + b_{2})$
		Step 2 输出 $μ$ 与 $log σ$：
			$μ      = tanh$？不需要，直接线性输出
			$log σ  = clamp(logstd\_head(h), LOG\_STD\_MIN, LOG\_STD\_MAX)$   
			$σ      = exp(log σ)$                                        
			clamp 范围一般 [-20, 2] 防止数值爆炸。
		Step 3 重参数化采样
			$ε - N(0, I)$                                 
			$z  = μ + σ  ε$              
			注意 z 是高斯变量，未经过 $tanh$。
		Step 4 tanh 压缩到动作空间
			$a = tanh(z)$
		Step 5 计算 $log π(a|s)$  
			由于 $tanh$ 不可逆，需要**变量替换公式**：
			$log π(a|s) = Σ_i [ log N(z_i | μ_i, σ_i²) - log(1 - a_i² + 1e-6) ]$ 
			其中
			$log N(z_i | μ_i, σ_i²) = -½log(2π) - log σ_i - ½((z_i - μ_i)/σ_i)²$ 
			1e-6 防止除零。因为 $tanh$ 单调，Jacobian 项就是 $-log(1 - tanh²(z)) = -log(1 - a²)$。
		Step 6 返回
			$return $a, log π(a|s)$ $		
			$a$ 用于与环境交互；$log π$ 用于策略梯度与熵正则。
	2. 训练时怎么用 $log\_prob$
		在策略梯度里，Actor 的损失为
		$L_{actor} = E_{s~D}[ E_{a~π}[ α log π_φ(a|s) - min(Q_{θ1}(s,a), Q_{θ2}(s,a)) ] ]$
		第一项 α log π 是熵正则，鼓励探索；
		第二项是 Critic 提供的 Q 值信号；
		由于 log_prob 已经可导，直接对 φ 反向传播即可。
	Actor 把状态 → 高斯参数 → 重参数化 → $tanh$ → 动作 + $log\_prob$，  
	log_prob 通过变量替换公式计入 Jacobian，保证整条链可导
****
- Critic（双Q网络）
	运算四阶段（公式 + 代码）
	    阶段 A：前向预测  
给定 (s,a) → Q_θ(s,a)
```python
q1 = critic1(s, a)    # [B,1]
q2 = critic2(s, a)    # [B,1]
```

阶段 B：计算目标值 y

1. 下一状态 s′ 由环境给出。
    
2. 用 **当前 Actor** 采样 a′ 及其 log-prob：
    

Python

复制

```python
a_next, log_prob_next = actor(s_next)   # [B, act_dim], [B,1]
```

3. 用 **目标网络** 计算 Q′(s′,a′)：
    

Python

复制

```python
q1_next = critic1_target(s_next, a_next)
q2_next = critic2_target(s_next, a_next)
```

4. 取最小 + 熵正则：
    

`y = r + γ * [ min(q1_next, q2_next) - α * log_prob_next ] (1)`

- α 是自适应温度；detach() 保证目标网络梯度不回传。
    
- r, γ 来自 Replay Buffer。
    

阶段 C：Critic 损失  
对两套参数分别做 MSE：

复制

```
L_Q1 = E_D [ (Q_θ1(s,a) – y.detach())² ]   (2)
L_Q2 = E_D [ (Q_θ2(s,a) – y.detach())² ]   (3)
```

Python

复制

```python
loss_q1 = (q1 - y.detach()).pow(2).mean()
loss_q2 = (q2 - y.detach()).pow(2).mean()
optimizer_c1.zero_grad(); loss_q1.backward(); optimizer_c1.step()
optimizer_c2.zero_grad(); loss_q2.backward(); optimizer_c2.step()
```

阶段 D：软更新目标网络（每步或每隔几步）

`θ_i′ ← τ θ_i + (1-τ) θ_i′ , τ ≈ 0.005 (4)`

Python

复制

```python
for param, targ_param in zip(critic1.parameters(), critic1_target.parameters()):
    targ_param.data.copy_(tau * param.data + (1-tau) * targ_param.data)
# 同理对 critic2_target
```

---

4. 维度与数值细节
    

- 动作 a 如果是连续多维，拼接后维度为 state_dim + action_dim。
    
- target 网络初始化用 `copy_` 与主网络完全相同；τ 很小保证慢跟踪。
    
- α 在目标计算里不需要梯度，因此 detach()；而在温度系数更新时 α 有梯度（见 Actor 部分）。
    

---

5. 一句话记忆
    

双 Critic 把 (s,a) → 标量 Q；  
目标值 = r + γ [ min(Q′) – α log π ]；  
两套网络分别最小化 MSE，再用软更新平滑目标。