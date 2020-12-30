import numpy as np
from tensorflow.python.types.core import Value
from src.data_input import DataSet
from enum import Enum


class NormalizationMethod(Enum):
    none = 0
    z_norm = 1
    min_max = 2
    mean_norm = 3
    vector_norm = 4


class Normalizer():
    def __init__(self, ds: DataSet, method: str) -> None:
        self.ds: DataSet = ds
        self.method = method
        self.descriptives = self.ds.descriptives()
        self.maxes_data = self.descriptives.get('data').get('max')
        self.mins_data = self.descriptives.get('data').get('min')
        self.means_data = self.descriptives.get('data').get('mean')
        self.stds_data = self.descriptives.get('data').get('std')

        self.max_label = self.descriptives.get('label').get('max')
        self.min_label = self.descriptives.get('label').get('min')
        self.mean_label = self.descriptives.get('label').get('mean')
        self.std_label = self.descriptives.get('label').get('std')

        self.labels = np.array(
            list(map(lambda x: x['label'], self.ds))).reshape(-1, 1)

        self.norm_label = self.labels / \
            np.math.sqrt(np.sum(self.labels*self.labels))

        if len(ds) < 2:  # unit testing...
            self.maxes_data = self.maxes_data.max()
            self.mins_data = self.mins_data.min()
            self.means_data = self.means_data.mean()
            self.stds_data = self.stds_data.std()

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
            return self.ds.data.copy(), self.labels
        else:
            raise ValueError(
                'Only (min_max, z_norm, mean_norm, vector_norm) are allowed.')

    def denormalize(self, data):
        if self.method == NormalizationMethod.min_max:
            return self._inverse_min_max(data)
        elif self.method == NormalizationMethod.z_norm:
            return self._inverse_z_norm(data)
        elif self.method == NormalizationMethod.mean_norm:
            return self._inverse_mean_norm(data)
        elif self.method == NormalizationMethod.vector_norm:
            return self._inverse_vector_norm(data)
        elif self.method == NormalizationMethod.none:
            return data
        else:
            raise ValueError(
                'Only (min_max, z_norm, mean_norm, vector_norm) are allowed.')

    def _inverse_min_max(self, data):
        return data*(self.max_label-self.min_label)+self.min_label

    def _inverse_z_norm(self, data):
        return data*self.stds_data + self.mean_label

    def _inverse_mean_norm(self, data):
        return data * (self.max_label-self.min_label) + self.mean_label

    def _inverse_vector_norm(self, data):
        if self.norm is not None:
            return data*self.norm
        else:
            raise ValueError(
                "You have not performed a vector normalization process.")

    def _min_max(self):
        data = self.ds.data.copy()
        data = (data - self.mins_data) / (self.maxes_data - self.mins_data)
        return data, self.labels

    def _z_norm(self):
        data = self.ds.data.copy()
        data = (data - self.means_data) / self.stds_data
        return data, self.labels

    def _mean_norm(self):
        data = self.ds.data.copy()
        data = (data - self.means_data) / (self.maxes_data - self.mins_data)
        return data, self.labels

    def _vector_norm(self):
        data = self.ds.data.copy()
        sq = data*data
        self.norm = np.math.sqrt(np.sum(sq))
        return data / self.norm, self.labels
