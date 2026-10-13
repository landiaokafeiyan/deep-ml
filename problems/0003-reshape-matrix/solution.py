import numpy as np

def reshape_matrix(
    a: list[list[int | float]],
    new_shape: tuple[int, int]
) -> list[list[int | float]]:

    if not a:
        return []

    if not new_shape:
        return []

    old_rows = len(a)
    old_cols = len(a[0])

    new_rows, new_cols = new_shape

    # Total number of elements must remain unchanged.
    if old_rows * old_cols != new_rows * new_cols:
        return []

    # Step 1: flatten the original matrix.
    all_elements = []

    for row in range(old_rows):
        for col in range(old_cols):
            all_elements.append(a[row][col])

    # Step 2: rebuild the matrix using the new shape.
    reshaped_matrix = []

    index = 0

    for row in range(new_rows):

        new_row = []

        for col in range(new_cols):
            new_row.append(all_elements[index])
            index += 1

        reshaped_matrix.append(new_row)

    return reshaped_matrix