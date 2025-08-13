[[network]]
- Critic（双Q网络）
	Critic 损失、Actor 损失和温度参数 α 的损失分别有不同的作用，它们共同推动智能体的学习过程，以实现高效、稳定的策略优化。
	Critic 损失的作用是**评估当前策略的性能**，并为 Actor 提供反馈信号。Critic 网络的目标是准确估计状态-动作对的 Q 值（即预期回报）。
		- **估计 Q 值**：Critic 网络通过学习预测在给定状态下采取某个动作的预期回报。它通过最小化预测 Q 值与目标 Q 值之间的误差来优化自身参数。
		- **提供目标信号**：Critic 的输出为 Actor 提供了目标信号，帮助 Actor 学习更优的策略。
	Critic 损失通常定义为： $L_Q​=E_{(s,a,r,s^′)∼D}​[(Q(s,a)−y)^2]$
		其中：Q(s,a) 是 Critic 网络预测的 Q 值。
			y 是目标 Q 值，计算公式为：
			$y=r+γ(1−d)[min(Q_{target​}(s^′,a^′)−αlogπ(a^′∣s^′)]$
			r 是即时奖励；γ 是折扣因子；d 是终止标志（是否到达终点）; $Q_{target​ }$是目标 Critic 网络的输出；α 是温度参数，用于控制熵的正则化；π(a′∣s′) 是策略网络（Actor）在状态 s′ 下采样出的动作 a′ 的概率。
	作用总结：
		- **优化 Critic 网络**：通过最小化预测 Q 值与目标 Q 值之间的误差，Critic 网络能够更准确地评估策略的性能。
		- **为 Actor 提供反馈**：Critic 的输出为 Actor 提供了关于当前策略优劣的反馈，帮助 Actor 学习更优的策略。
		
	
- 运算四阶段（公式 + 代码）
	阶段 A：前向预测  
		给定 (s,a) → Q_θ(s,a)
		`q1 = critic1(s, a)    # [B,1]`
		`q2 = critic2(s, a)    # [B,1]`
	阶段 B：计算目标值 y
		1. 下一状态 s′ 由环境给出。
		2. 用 **当前 Actor** 采样 a′ 及其 log-prob：
			`a_next, log_prob_next = actor(s_next)   # [B, act_dim], [B,1]`
		3. 用 **目标网络** 计算 Q′(s′,a′)：
			`q1_next = critic1_target(s_next, a_next)`
			`q2_next = critic2_target(s_next, a_next)`
		4. 取最小 + 熵正则：
			`y = r + γ * [ min(q1_next, q2_next) - α * log_prob_next ] (1)`
			- α 是自适应温度；detach() 保证目标网络梯度不回传。
			- r, γ 来自 Replay Buffer。
	阶段 C：Critic 损失  
		对两套参数分别做 MSE：
		`L_Q1 = E_D [ (Q_θ1(s,a) – y.detach())² ]   (2)`
		`L_Q2 = E_D [ (Q_θ2(s,a) – y.detach())² ]   (3)
	`loss_q1 = (q1 - y.detach()).pow(2).mean()`
		`loss_q2 = (q2 - y.detach()).pow(2).mean()`
		`optimizer_c1.zero_grad(); loss_q1.backward(); optimizer_c1.step()`
		`optimizer_c2.zero_grad(); loss_q2.backward(); optimizer_c2.step()`
	阶段 D：软更新目标网络（每步或每隔几步）
		`θ_i′ ← τ θ_i + (1-τ) θ_i′ , τ ≈ 0.005 (4)`
		`for param, targ_param in zip(critic1.parameters(), critic1_target.parameters()):`
			`targ_param.data.copy_(tau * param.data + (1-tau) * targ_param.data)`
			`#同理对 critic2_target`

- 维度与数值细节
	1. 动作 a 如果是连续多维，拼接后维度为 state_dim + action_dim。
	2. target 网络初始化用 `copy_` 与主网络完全相同；τ 很小保证慢跟踪。
	3. α 在目标计算里不需要梯度，因此 detach()；而在温度系数更新时 α 有梯度（见 Actor 部分）。
- 一句话记忆
	双 Critic 把 (s,a) → 标量 Q；  
	目标值 = r + γ [ min(Q′) – α log π ]；  
	两套网络分别最小化 MSE，再用软更新平滑目标。