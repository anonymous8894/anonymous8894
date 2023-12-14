import numpy as np


class SplitUtil:
    def __init__(self, minval, maxval, num_splits):
        self.minval = minval
        self.num_splits = num_splits
        self.split_size = (maxval - minval) / num_splits

    def get_split(self, val):
        result = (val - self.minval) / self.split_size
        result = np.floor(result)
        result = np.clip(result, 0, self.num_splits - 1)
        return result
