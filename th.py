import numpy as np

def is_subarray(x, y):
    # Length of x and y
    len_x, len_y = len(x), len(y)
    
    # If y is longer than x, it cannot be a subarray
    if len_y > len_x:
        return False

    # Create a sliding window view of x
    subarrays = np.lib.stride_tricks.sliding_window_view(x, len_y)

    # Check if any of the subarrays match y
    return np.any(np.all(subarrays == y, axis=1))