import numpy as np

def impute_missing_data(
    data: np.ndarray,
    strategy: str = "mean"
) -> np.ndarray:

    # Work on a copy so we do not modify the original input.
    result = data.copy().astype(float)

    rows, cols = result.shape

    for col in range(cols):

        # Extract one column.
        column = result[:, col]

        # Keep only non-missing values.
        valid_values = column[~np.isnan(column)]

        # If the whole column is missing,
        # there is no valid statistic to compute.
        if len(valid_values) == 0:
            continue

        if strategy == "mean":
            fill_value = np.mean(valid_values)

        elif strategy == "median":
            fill_value = np.median(valid_values)

        elif strategy == "mode":
            values, counts = np.unique(
                valid_values,
                return_counts=True
            )

            # np.unique returns sorted values.
            # np.argmax returns the first max count,
            # so ties automatically choose the smallest value.
            fill_value = values[np.argmax(counts)]

        else:
            raise ValueError(
                "strategy must be 'mean', 'median', or 'mode'"
            )

        # Find missing positions in this column.
        missing_mask = np.isnan(column)

        # Replace NaN with the chosen fill value.
        result[missing_mask, col] = fill_value

    return result