import numpy as np
from itertools import product
from tqdm import tqdm
from joblib import delayed, Parallel

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

def find_in_epochs(template, data):
    assert template.shape[2] >= data.shape[2]
    raw = np.zeros((data.shape[0], ), dtype=bool)
    idx = 0

    def _match_to_template(d):
        for cur_template in template:
            matches_first = is_subarray(d[0], cur_template[0])
            if matches_first:
                matches = [is_subarray(d, t) for d, t in zip(d, cur_template)]
                if np.all(matches):
                    return True
            
        return False

    raw = Parallel(n_jobs=-1, verbose=5)(delayed(_match_to_template)(d) for d in data)
    
    # for cur_data in tqdm(data):
    #     for cur_template in template:
    #         matches = [is_subarray(d, t) for d, t in zip(cur_data, cur_template)]
    #         raw[idx] = np.all(matches)
    #         if raw[idx]:
    #                continue
    #     idx += 1
    
    return raw