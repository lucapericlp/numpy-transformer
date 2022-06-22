import numpy as np

def softmax(y: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    softmax a tensor with row-wise being the default.
    """
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    return y / ax_sum
