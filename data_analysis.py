from data_input import CSVFile, CSVReader, DataSet


class DataAnalysis(object):
    def __init__(self, ds: DataSet) -> None:
        super().__init__()
        self.ds = ds

    def std_intervals(self, sigma: int):
        if not 1 <= sigma:
            raise ValueError('Sigma is >= 1.')

        descriptives = self.ds.descriptives()

        intervals_for_features = []

        for i, desc in enumerate(descriptives):
            data = desc.get('data')
            mean = data.get('mean')
            sd = data.get('std')

            intervals = [(-sd*n + mean, sd*n + mean) for n in sigma]

            intervals_for_features.append(
                {
                    'feature_index': i,
                    'intervals': intervals
                }
            )


if __name__ == "__main__":
    reader = CSVReader(
        [CSVFile(fn='iris.csv', delimiter=',', header=True, to_numeric=True, one_hot_classes=3), CSVFile(fn='linnerud_exercise.csv', delimiter=' ', header=True, to_numeric=True)])
    hey = reader.read_csv()
    for csv in hey:
        print(csv.descriptives())
        print(csv.size())

    analysis = DataAnalysis(hey[0])
