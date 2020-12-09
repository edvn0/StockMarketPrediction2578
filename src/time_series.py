from src.normalization import NormalizationMethod, Normalizer
from typing import List
import numpy as np
from src.data_input import DataEntry, DataSet


class TimeSeries(object):
    def __init__(self, ds: DataSet,  window_size: int, label_size: int, normalization_method: NormalizationMethod) -> None:
        super().__init__()
        self.window_size = window_size
        self.label_size = label_size
        self.total_size = self.window_size + self.label_size
        self.ds = ds

        self.normalizer = Normalizer(self.ds, normalization_method)
        self.normed_data = self.normalizer.normalize()

        self.size = self.ds.size()[0]

    def generate(self, split=True):
        time_series: List[DataEntry] = []
        step = self.window_size
        k = 0
        data = self.normed_data
        label = np.array(
            list(map(lambda x: x['label'], self.ds))).reshape(-1, 1)
        for i in range(0, self.size, step):

            window_data = data[i:i+self.window_size]
            window_label = label[i+self.window_size: i +
                                 self.window_size+self.label_size]
            k += 1
            entry = DataEntry(window_data, window_label,  k)
            time_series.append(entry)

        # Check if the last time series step does not fit, remove
        data_last = time_series[-1]
        if len(data_last.data) != self.window_size or not data_last.label:
            time_series = time_series[:-1]

        ds = DataSet(time_series)
        if split:
            return ds.split_data()
        return ds
