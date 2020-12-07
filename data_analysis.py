import numpy as np
from data_input import CSVFile, CSVReader, DataSet


class DataAnalysis(object):
    def __init__(self, ds: DataSet) -> None:
        super().__init__()
        self.ds = ds

    def std_intervals(self, sigma: int):
        if not 1 <= sigma:
            raise ValueError('Sigma is >= 1.')

        descriptives = self.ds.descriptives()

        intervals = []

        for sub in descriptives:
            dataset_metrics = descriptives.get(sub).values()
            values = np.array([x for x in dataset_metrics])

            mean = values[0]
            var = values[1]
            sd = values[2]

            hi = [mean + sd * n for n in range(sigma)]
            lo = [mean - sd * n for n in range(sigma)]

            interval = [(lo[i], hi[i]) for i in range(len(dataset_metrics))]

            print("hi", hi)
            print("lo", lo)

            intervals.append(interval)
        return intervals


if __name__ == "__main__":
    reader = CSVReader(
        [CSVFile(fn='iris.csv', delimiter=',', header=True, to_numeric=True)])
    hey = reader.read_csv()
    for csv in hey:
        print(csv.descriptives())
        print(csv.size())

    analysis = DataAnalysis(hey[0])
    print(analysis.std_intervals(sigma=3))
