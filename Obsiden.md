$$
\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 & 0 & 0 \\
\mu_1 & 2 & \lambda_1 & \cdots & 0 & 0 & 0 \\
0 & \mu_2 & 2 & \cdots & 0 & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & \mu_{n-1} & 2 & \lambda_{n-1} \\
0 & 0 & 0 & \cdots & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
M_0 \\ M_1 \\ M_2 \\ \vdots \\ M_{n-1} \\ M_n
\end{bmatrix}
=
\begin{bmatrix}
0 \\ d_1 \\ d_2 \\ \vdots \\ d_{n-1} \\ 0
\end{bmatrix}
$$$h_i = x_{i+1} - x_i$
$\mu_i = \frac{h_{i-1}}{h_{i-1} + h_i}$
$\lambda_i = 1 - \mu_i$
$d_i = 6 \cdot \frac{\frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}}{h_{i-1} + h_i}$

