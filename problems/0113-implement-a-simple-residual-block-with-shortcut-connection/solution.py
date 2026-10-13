# import numpy as np

# def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
# 	# Your code here
# #Residual connections provide an identity path for both activations and gradients. Instead of learning the full mapping directly, the block learns a residual function \(F(x)\), which makes optimization easier and helps mitigate degradation and vanishing-gradient problems in very deep networks

import numpy as np

def residual_block(
    x: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray
) -> np.ndarray:

    # First linear layer
    h1 = w1 @ x

    # ReLU
    h1 = np.maximum(0, h1)

    # Second linear layer
    h2 = w2 @ h1

    # ReLU
    h2 = np.maximum(0, h2)

    # Shortcut connection
    output = h2 + x

    # Final ReLU
    output = np.maximum(0, output)

    return output