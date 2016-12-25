import numbers
import numpy as np
from numpy import float32


def features_scaling(datasets):
    if not isinstance(datasets, list):
        raise ValueError
    for data in datasets:
        if not isinstance(data, list):
            raise ValueError
        for value in data:
            if not isinstance(value, numbers.Number):
                raise ValueError

    scaled_datasets = np.array(datasets, copy=True, dtype=float32)
    for i in range(scaled_datasets.shape[1]):
        mean = np.mean(scaled_datasets[:, i])
        scaled_datasets[:, i] = scaled_datasets[:, i] / mean

    return scaled_datasets.tolist()
