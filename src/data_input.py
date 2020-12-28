import csv as csv
from numbers import Number
import os
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.python.data.ops.dataset_ops import Dataset


class DataEntry():
    def __init__(self, data, label, row_index, meta=None):
        self.data = data
        self.label = label
        self.row_index = row_index
        self.meta = meta

    def __getitem__(self, item):
        if item == 'data':
            return self.data
        elif item == 'label':
            return self.label
        elif item == 'index':
            return self.row_index
        elif item == 'meta':
            return self.meta
        else:
            raise ValueError(
                'Only data, label, index, and meta are allowed as indices into this object.')

    @property
    def feature_size(self):
        return len(self.data)

    def __repr__(self) -> str:
        return ", ".join([f'Data: {self.data}', f'Label: {self.label}'])


class DataSet():
    def __init__(self, data_list: List[DataEntry], identifier=None) -> None:
        self.ds = data_list
        self.id = identifier
        self.data: np.ndarray = np.array(
            list(map(lambda x: x['data'], self.ds)), dtype=np.double)
        self.labels: np.ndarray = np.array(
            list(map(lambda x: x['label'], self.ds)), dtype=np.double)
        self.classes = np.shape(self.labels[0])
        self.indicies: np.ndarray = np.array(
            list(map(lambda x: x['index'], self.ds)), dtype=np.double).reshape(-1, 1)
        self.features = self.ds[0].feature_size

    def identifier(self):
        return self.id

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

    def split_data(self):
        return self.data, self.labels


class CSVFile(object):
    def __init__(self, fn: str, delimiter: str, header: bool, to_numeric: bool, one_hot_classes: int = 0,
                 prefix=None) -> None:
        super().__init__()
        self.prefix = prefix
        self.filename = fn if self.prefix is None else os.path.join(
            self.prefix, fn)
        self.delimiter = delimiter
        self.header = header,
        self.to_numeric = to_numeric
        self.one_hot = one_hot_classes > 0
        self.classes = one_hot_classes


class CSVReader():
    def __init__(self, data_indices: List[int], label_index: int, info_indices: list, filenames: List[CSVFile] = None,
                 dir: str = None) -> None:
        """Creates a CSV reader for Stock Market historic files.

        Args:
            header_indices (List[int]): which headers do you need to extract, by indicies, example:
                    "Name", "Age", "Weight", "Height", "Gender", "date_created" -> [0,1,2,3,4] indicating you do not need date_created
            filenames (List[CSVFile], optional): List of CSVFiles to analyse. Defaults to None.
            dir (str, optional): Directory of csv files to gather. Defaults to None.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        self.data_indices = data_indices
        self.label_index = label_index
        self.from_dir = False
        self.info_indices = info_indices
        if dir is not None and filenames is not None:
            raise ValueError(
                'Choose either a directory, or create your own files in a list.')
        elif dir is not None:
            self.dir = dir
            self.files = os.listdir(self.dir)
            self.files = list(filter(lambda x: '.csv' in str(x), self.files))
            self.from_dir = True
        elif filenames is not None:
            self.files = filenames
        else:
            raise ValueError(
                'Choose either a directory, or create your own files in a list.')

    def read_csv(self) -> List[DataSet]:
        if self.from_dir:
            return self._read_from_dir()
        else:
            return self._read_from_csv_file()

    def _read_from_dir(self):
        files = [CSVFile(fn, delimiter=',', header=True, to_numeric=True, prefix=self.dir)
                 for fn in self.files]
        return self._files_to_csv(files)

    def _read_from_csv_file(self):
        return self._files_to_csv(self.files)

    def _files_to_csv(self, files: List[CSVFile]):
        output_csv: List[DataSet] = []
        for f in files:
            with open(f.filename, mode='r') as csv_file:
                reader = csv.reader(csv_file, delimiter=f.delimiter)
                to_csv_file = []
                if f.header:
                    next(reader)

                for i, row in enumerate(reader):
                    row_size = len(row)
                    if row_size == 0:
                        continue

                    if 'null' in row:
                        continue

                    # reasonable assumption for classification
                    data = []
                    for d in self.data_indices:
                        data.append(row[d])
                    # might be onehot, you have to solve this yourself.

                    label = row[self.label_index]

                    info = [row[d] for d in self.info_indices]

                    if f.to_numeric:  # force numeric if not already...
                        data = list(map(float, data))
                        label = float(label)

                    if f.one_hot:
                        label_to_int = int(label)
                        one_hot = [0 for _ in range(f.classes)]
                        one_hot[label_to_int] = 1
                        label = one_hot

                    entry = DataEntry(data, label, i, info)
                    to_csv_file.append(entry)

                name = os.path.basename(csv_file.name)
                output_csv.append(DataSet(to_csv_file, name))

        return output_csv
