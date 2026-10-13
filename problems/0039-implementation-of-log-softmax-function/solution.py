import numpy as np
import numpy as np

def log_softmax(scores: list) -> np.ndarray:

    if not scores:
        return np.array([])

    max_value = np.max(scores)

    exp_sum = 0.0

    for score in scores:
        exp_sum += np.exp(score - max_value)

    log_sum_exp = np.log(exp_sum)

    result = []

    for score in scores:
        value = (
            score - max_value
            - log_sum_exp
        )

        result.append(value)

    return np.array(result)
def log_softmax(scores: list) -> np.ndarray:
    """
    Compute numerically stable LogSoftmax.

    LogSoftmax(z_i)
        = z_i - log(sum_j exp(z_j))

    Stable version:
        = (z_i - max(z))
          - log(sum_j exp(z_j - max(z)))
    """

    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        return np.array([])

    # Step 1: subtract max for numerical stability
    max_value = np.max(scores)

    shifted_scores = scores - max_value

    # Step 2: compute log-sum-exp
    log_sum_exp = np.log(
        np.sum(np.exp(shifted_scores))
    )

    # Step 3: log softmax
    result = shifted_scores - log_sum_exp

    return result