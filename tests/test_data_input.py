import unittest
import numpy as np
from src.data_input import DataEntry, DataSet

from src.normalization import NormalizationMethod, Normalizer


class DataInputTest(unittest.TestCase):

    def test_norming_data_vector_norm(self):
        ds = DataSet([DataEntry([1, 1, 2], 0, 0)])
        normer = Normalizer(ds, NormalizationMethod.vector_norm)
        self.assertTrue(np.allclose(
            np.array([[1, 1, 2]])/2.4494897427832, normer._vector_norm()))

    def test_norming_data_min_max(self):
        ds = DataSet([DataEntry([1, 1, 2], 0, 0)])
        normer = Normalizer(ds, NormalizationMethod.min_max)
        data = np.array([1, 1, 2])
        normed = (data - np.min(data)) / (np.max(data)-np.min(data))
        self.assertTrue(np.allclose(normed, normer.normalize()))


if __name__ == 'main':
    unittest.main()
