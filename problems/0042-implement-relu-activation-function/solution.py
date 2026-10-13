def relu(z: float) -> float:
	# Your code here
	return z if z>0 else 0
#ReLU 最重要的面试点是它为什么曾经比 sigmoid 更适合作为 hidden-layer activation：正区间梯度是 1，不容易像 sigmoid 那样在大 \(|z|\) 时因为 saturation 导致梯度接近 0；同时计算非常便宜