import math

def softmax(scores: list[float]) -> list[float]:
    """
    Compute numerically stable softmax.

    Args:
        scores: list of logits

    Returns:
        list of probabilities summing to 1
    """

    if not scores:
        return []

    # Step 1:
    # subtract max for numerical stability
    max_score = max(scores)

    # Step 2:
    # exponentiate shifted logits
    exp_scores = []

    for score in scores:
        exp_scores.append(
            math.exp(score - max_score)
        )

    # Step 3:
    # normalization denominator
    total = sum(exp_scores)

    # Step 4:
    # convert to probabilities
    probabilities = []

    for value in exp_scores:
        probabilities.append(
            value / total
        )

    return probabilities