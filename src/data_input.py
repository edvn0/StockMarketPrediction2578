from typing import Dict, List, Tuple
import numpy as np


class DataEntry(object):
    def __init__(self, data, label, row_index) -> None:
        super().__init__()
        self.data = data
        self.label = label
        self.row_index = row_index

    def __getitem__(self, item):
        if item == 'data':
            return self.data
        elif item == 'label':
            return self.label
        elif item == 'index':
            return self.row_index
        else:
            raise ValueError(
                'Only data, label and index are allowed as indices into this object.')

    @property
    def feature_size(self):
        return len(self.data)

    def __repr__(self) -> str:
        return ", ".join([f'Data: {self.data}', f'Label: {self.label}'])


class DataSet(object):
    def __init__(self, data_list: List[DataEntry]) -> None:
        super().__init__()
        self.ds = data_list
        self.data: np.ndarray = np.array(
            list(map(lambda x: x['data'], self.ds)))
        self.labels: np.ndarray = np.array(
            list(map(lambda x: x['label'], self.ds)))
        self.classes = np.shape(self.labels[0])
        self.indicies: np.ndarray = np.array(
            list(map(lambda x: x['index'], self.ds))).reshape(-1, 1)
        self.features = self.ds[0].feature_size

    def size(self) -> Tuple:
        return (len(self), self.features, self.classes)

    def descriptives(self) -> Dict[str, Dict[str, np.ndarray]]:
        mean_data = self.data.mean(axis=0)
        var_data = self.data.var(axis=0)
        std_data = self.data.std(axis=0)
        max_data = self.data.max(axis=0)
        min_data = self.data.min(axis=0)

        mean_label = self.labels.mean(axis=0)
        var_label = self.labels.var(axis=0)
        std_label = self.labels.std(axis=0)
        max_label = self.labels.max(axis=0)
        min_label = self.labels.min(axis=0)

        return {
            'data':
            {
                'mean': mean_data,
                'var': var_data,
                'std': std_data,
                'max': max_data,
                'min': min_data
            },
            'label':
            {
                'mean': mean_label,
                'var': var_label,
                'std': std_label,
                'max': max_label,
                'min': min_label
            }
        }

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        return self.ds[index]

    def get_range(self, start: int, end: int):
        return self.ds[start:end]

    def sample(self, size: int):
        return np.random.choice(self.ds, size, replace=True)

    def __repr__(self) -> str:
        return str(self.ds)


class CSVFile(object):
    def __init__(self, fn: str, delimiter: str, header: bool, to_numeric: bool, one_hot_classes: int = 0) -> None:
        super().__init__()
        self.filename = fn
        self.delimiter = delimiter
        self.header = header,
        self.to_numeric = to_numeric
        self.one_hot = one_hot_classes > 0
        self.classes = one_hot_classes


class CSVReader(object):
    def __init__(self, filenames: List[CSVFile]) -> None:
        super().__init__()
        self.files = filenames

    def read_csv(self) -> List[DataSet]:
        import csv as csv

        files_to_csv: List[DataSet] = []

        for f in self.files:
            with open(f.filename, mode='r') as csv_file:
                reader = csv.reader(csv_file, delimiter=f.delimiter)
                to_csv_file = []
                if f.header:
                    next(reader)

                for i, row in enumerate(reader):
                    row_size = len(row)
                    # reasonable assumption for classification
                    data = row[:row_size-1]
                    # might be onehot, you have to solve this yourself.
                    label = row[row_size-1]

                    if f.to_numeric:  # force numeric if not already...
                        data = list(map(float, data))
                        label = float(label)

                    if f.one_hot:
                        label_to_int = int(label)
                        one_hot = [0 for _ in range(f.classes)]
                        one_hot[label_to_int] = 1
                        label = one_hot

                    entry = DataEntry(data, label, i)
                    to_csv_file.append(entry)

                files_to_csv.append(DataSet(to_csv_file))

        return files_to_csv
