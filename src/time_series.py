from src.normalization import NormalizationMethod, Normalizer
from typing import List
import numpy as np
from src.data_input import DataEntry, DataSet


class TimeSeries(object):
    def __init__(self, normer: Normalizer, window_size: int, label_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.label_size = label_size

        self.normalizer = normer
        self.normed_data = self.normalizer.normalize()

        self.size = self.normalizer.ds.size()[0]

    def generate(self, split=True):
        time_series: List[DataEntry] = []
        k = 0
        data, labels = self.normed_data
        print(data.shape)
        for i in range(self.window_size, self.size):

            window_data = data[i-self.window_size:i]
            window_label = labels[i]
            k += 1
            entry = DataEntry(window_data, window_label,  k)
            time_series.append(entry)

        # Check if the last time series step does not fit, remove
        data_last = time_series[-1]
        if len(data_last.data) != self.window_size or not data_last.label:
            time_series = time_series[:-1]

        ds = DataSet(time_series)
        exit()
        if split:
            return ds.split_data()
        return ds
