def min_max(x: list[float]) -> list[float]:
    """
    Scale values to [0, 1].

    Formula:
        x_scaled = (x - min_x) / (max_x - min_x)
    """

    if not x:
        return []

    min_val = min(x)
    max_val = max(x)

    # Edge case:
    # all values are identical, denominator = 0
    if max_val == min_val:
        return [0.0] * len(x)

    y = []

    for value in x:
        scaled_value = (
            value - min_val
        ) / (
            max_val - min_val
        )

        y.append(scaled_value)

    return y