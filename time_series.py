from typing import List
import numpy as np
from data_input import CSVFile, CSVReader, DataEntry, DataSet
import math


class TimeSeries(object):
    def __init__(self, ds: DataSet,  window_size: int, label_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.label_size = label_size
        self.total_size = self.window_size + self.label_size
        self.ds = ds
        self.size = self.ds.size()[0]

    def generate(self):
        time_series: List[DataEntry] = []
        step = self.total_size
        print("Step size", step)
        k = 0
        data = np.array(
            list(map(lambda x: x['data'], self.ds))).reshape(-1, 1)
        label = np.array(
            list(map(lambda x: x['label'], self.ds))).reshape(-1, 1)
        for i in range(0, self.size, step):

            window_data = data[i:i+step]
            window_label = label[i+self.window_size: i +
                                 self.window_size+self.label_size]
            k += 1
            entry = DataEntry(window_data, window_label,  k)
            time_series.append(entry)

        # Check if the last time series step does not fit, remove
        data_last = time_series[-1]
        if len(data_last.data) != step or not data_last.label:
            time_series = time_series[:-1]

        return DataSet(time_series)
