def scalar_multiply(
    matrix: list[list[int | float]],
    scalar: int | float
) -> list[list[int | float]]:

    if not matrix:
        return []

    new_matrix = []

    for i in range(len(matrix)):
        new_row = []

        for j in range(len(matrix[0])):
            new_row.append(matrix[i][j] * scalar)

        new_matrix.append(new_row)

    return new_matrix