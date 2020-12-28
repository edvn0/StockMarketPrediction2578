import numpy as np
from src.data_input import DataSet
from enum import Enum


class NormalizationMethod(Enum):
    none = 0
    z_norm = 1
    min_max = 2
    mean_norm = 3
    vector_norm = 4


class Normalizer(object):
    def __init__(self, ds: DataSet, method: str) -> None:
        super().__init__()
        self.ds = ds
        self.method = method
        self.descriptives = self.ds.descriptives()
        self.maxes_data = self.descriptives.get('data').get('max')
        self.mins_data = self.descriptives.get('data').get('min')
        self.means_data = self.descriptives.get('data').get('mean')
        self.stds_data = self.descriptives.get('data').get('std')

    def normalize(self):
        if self.method == NormalizationMethod.min_max:
            return self._min_max()
        elif self.method == NormalizationMethod.z_norm:
            return self._z_norm()
        elif self.method == NormalizationMethod.mean_norm:
            return self._mean_norm()
        elif self.method == NormalizationMethod.vector_norm:
            return self._vector_norm()
        elif self.method == NormalizationMethod.none:
            return self.ds.data.copy()
        else:
            raise ValueError('Only (min_max, z_norm) are allowed.')

    def _min_max(self):
        data = self.ds.data.copy()
        data = (data - self.mins_data) / (self.maxes_data - self.mins_data)
        return data

    def _z_norm(self):
        data = self.ds.data.copy()
        data = (data - self.means_data) / self.stds_data
        return data

    def _mean_norm(self):
        data = self.ds.data.copy()
        data = (data - self.means_data) / (self.maxes_data - self.mins_data)
        return data

    def _vector_norm(self):
        data = self.ds.data.copy()
        sq = data*data
        norm = np.math.sqrt(np.sum(sq))
        return data / norm
