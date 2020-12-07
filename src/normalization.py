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
        self.maxes = self.descriptives.get('data').get('max')
        self.mines = self.descriptives.get('data').get('min')
        self.means = self.descriptives.get('data').get('mean')
        self.stds = self.descriptives.get('data').get('std')

    def normalize(self):
        if self.method == NormalizationMethod.min_max:
            return self._min_max()
        elif self.method == NormalizationMethod.z_norm:
            return self._z_norm()
        elif self.method == NormalizationMethod.mean_norm:
            return self._mean_norm()
        elif self.method == NormalizationMethod.vector_norm:
            return self._vector_norm()
        elif self.methoid == NormalizationMethod.none:
            return self.ds.data.copy()
        else:
            raise ValueError('Only (min_max, z_norm) are allowed.')

    def _min_max(self):
        data = self.ds.data.copy()
        data = (data - self.mines) / (self.maxes - self.mines)
        return data

    def _z_norm(self):
        data = self.ds.data.copy()
        print(self.means)
        print(self.stds)
        data = (data - self.means) / (self.stds)
        return data

    def _mean_norm(self):
        data = self.ds.data.copy()
        data = (data - self.means) / (self.maxes - self.mines)
        return data

    def _vector_norm(self):
        data = self.ds.data.copy()
        sq = data*data
        norm = np.math.sqrt(np.sum(sq))
        return data / norm
