import math

def elu(x: float, alpha: float = 1.0) -> float:
    val = x if x > 0 else alpha * (math.exp(x) - 1)
    return round(val, 4)