import math

def binary_cross_entropy(
    y_true: list,
    y_pred: list,
    epsilon: float = 1e-15
) -> float:

    total_loss = 0.0

    for y, p in zip(y_true, y_pred):

        # Clip probability
        p = max(
            epsilon,
            min(p, 1 - epsilon)
        )

        loss = -(
            y * math.log(p)
            +
            (1 - y) * math.log(1 - p)
        )

        total_loss += loss

    return total_loss / len(y_true)