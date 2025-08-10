[[network]]
- Critic（双Q网络）
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