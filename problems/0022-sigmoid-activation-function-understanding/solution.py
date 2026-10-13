import math

def sigmoid(z: float) -> float:

    if z >= 0:
        result = 1 / (1 + math.exp(-z))

    else:
        exp_z = math.exp(z)
        result = exp_z / (1 + exp_z)

    return round(result, 4)