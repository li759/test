[[network]]
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