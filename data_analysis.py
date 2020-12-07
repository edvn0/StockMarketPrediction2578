from typing import List
import numpy as np
from data_input import DataSet


class DataAnalysis(object):
    def __init__(self, ds: DataSet) -> None:
        super().__init__()
        self.ds = ds

    def std_intervals(self, sigma: int):
        """Generates intervals of sigma significance around the mean for all features of this class

        Args:
            sigma (int): how many standard deviations do we accept

        Raises:
            ValueError: if sigma is < 1.

        Returns:
            List[Tuple[float, float]]: List of (mean - sd*k, mean + sd*k) for k = sigma.
        """
        if not 1 <= sigma:
            raise ValueError('Sigma is >= 1.')

        descriptives = self.ds.descriptives()

        intervals = []

        for sub in descriptives:
            dataset_metrics = descriptives.get(sub).values()
            values = np.array([x for x in dataset_metrics])

            mean = values[0]
            sd = values[2]

            hi = [mean + sd * n for n in range(sigma)]
            lo = [mean - sd * n for n in range(sigma)]

            interval = [(lo[i], hi[i]) for i in range(len(dataset_metrics))]

            intervals.append(interval)
        return intervals
