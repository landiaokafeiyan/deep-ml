def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z > 0 else alpha * z
#Leaky ReLU modifies ReLU by keeping a small non-zero slope in the negative region, which helps reduce the dying ReLU problem.