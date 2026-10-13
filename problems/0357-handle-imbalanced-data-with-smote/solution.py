import numpy as np

def smote(
    X_minority: np.ndarray,
    n_synthetic: int,
    k: int = 5
) -> np.ndarray:
    """
    Generate synthetic minority samples using SMOTE.

    Args:
        X_minority:
            2D array of shape [n_samples, n_features]

        n_synthetic:
            Number of synthetic samples to generate

        k:
            Number of nearest neighbors

    Returns:
        2D NumPy array of shape
        [n_synthetic, n_features]
    """

    X_minority = np.asarray(X_minority, dtype=float)

    n_samples, n_features = X_minority.shape

    # We cannot have more neighbors than
    # the number of other minority samples.
    k_actual = min(k, n_samples - 1)

    # Required edge cases.
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features))

    synthetic_samples = []

    for _ in range(n_synthetic):

        # ---------------------------------
        # Step 1:
        # randomly select base sample
        # ---------------------------------
        i = np.random.randint(0, n_samples)

        x_i = X_minority[i]

        # ---------------------------------
        # Step 2:
        # compute Euclidean distances
        # from x_i to every minority sample
        # ---------------------------------
        differences = X_minority - x_i

        distances = np.sqrt(
            np.sum(
                differences ** 2,
                axis=1
            )
        )

        # Sort indices by ascending distance.
        sorted_indices = np.argsort(distances)

        # The first element is x_i itself
        # because its distance is 0.
        nearest_indices = sorted_indices[1:k_actual + 1]

        # ---------------------------------
        # Step 3:
        # randomly select one neighbor
        # ---------------------------------
        j = np.random.randint(0, k_actual)

        neighbor_index = nearest_indices[j]

        x_nn = X_minority[neighbor_index]

        # ---------------------------------
        # Step 4:
        # interpolate
        # ---------------------------------
        gap = np.random.random()

        x_synthetic = (
            x_i
            +
            gap * (x_nn - x_i)
        )

        synthetic_samples.append(
            x_synthetic
        )

    return np.array(synthetic_samples)