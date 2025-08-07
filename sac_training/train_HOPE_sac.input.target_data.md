[[train_HOPE_sac.input]]
1. 环境 reset/step/render 时，调用 CarParking.render()。
2. render() 调用 _get_targt_repr()。
3. _get_targt_repr() 计算车辆与目标的相对距离、角度、朝向，生成长度为5的 numpy 数组。
4. 该数组作为 obs['target']，供后续神经网络输入。
- <font color="#ffffff">车辆当前位置和朝向</font>：($x_{ego},y_{ego},θ_{ego}$)
- 目标位置和朝向：($x_{dest},y_{dest},θ_{dest}$)
则 target 的五个分量为：
1. 相对距离
    $rel\_distance=\sqrt{(x_{dest}−x_{ego})^2+(y_{dest}−y_{ego})^2}$
2. 相对角度的余弦
    $rel\_angle=arctan⁡2(y_{dest}−y_{ego},x_{dest}−x_{ego})−θ_{ego}​$
    $cos⁡(rel\_angle)$
3. 相对角度的正弦
    $sin⁡(rel\_angle)$
4. 目标朝向与自身朝向的余弦
    $rel\_dest\_heading=θ_{dest}−θ_{ego}$
    $cos⁡(rel\_dest\_heading)$
5. 目标朝向与自身朝向的余弦（注意源码写了两次，可能是冗余或历史遗留）
    $cos⁡(rel\_dest\_heading)$

| 分量  | 公式                    | 物理意义               |
| --- | --------------------- | ------------------ |
| 0   | rel_distance          | 车辆当前位置到目标点的欧氏距离（米） |
| 1   | cos(rel_angle)        | 车辆面向方向与目标方向夹角的余弦   |
| 2   | sin(rel_angle)        | 车辆面向方向与目标方向夹角的正弦   |
| 3   | cos(rel_dest_heading) | 车辆朝向与目标朝向夹角的余弦     |
| 4   | cos(rel_dest_heading) | 同上，可能是冗余           |
