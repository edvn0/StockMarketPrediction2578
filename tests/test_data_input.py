import unittest

import numpy as np

from normalization import Normalizer


class DataInputTest(unittest.TestCase):

    def test_norming_data(self):
        normer = Normalizer()
        self.assertAlmostEqual(np.array([1, 1, 2])/2.4494897427832,
                               normer._vector_norm(np.array([1, 1, 2])))

    pass


if __name__ == 'main':
    unittest.main()
