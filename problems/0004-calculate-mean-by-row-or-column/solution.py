def calculate_matrix_mean(
    matrix: list[list[float]],
    mode: str
) -> list[float]:

    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    if mode == "row":
        result = []

        for row in matrix:
            row_mean = sum(row) / cols
            result.append(row_mean)

        return result

    if mode == "column":
        result = []

        for col in range(cols):
            column_sum = 0

            for row in range(rows):
                column_sum += matrix[row][col]

            column_mean = column_sum / rows
            result.append(column_mean)

        return result

    return []