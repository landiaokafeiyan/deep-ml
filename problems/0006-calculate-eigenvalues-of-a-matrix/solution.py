import numpy as np

def calculate_eigenvalues(
    matrix: list[list[float | int]]
) -> list[float]:

    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    if rows != cols:
        raise ValueError("Matrix must be square.")

    A = np.array(matrix, dtype=float)

    eigenvalues = np.linalg.eigvals(A)

    return eigenvalues.tolist()