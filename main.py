import numpy as np
from src.data_prediction import tf_model
from src.normalization import NormalizationMethod
from src.data_input import CSVReader, CSVFile
from src.data_analysis import DataAnalysis
from src.time_series import TimeSeries
from sklearn.metrics import multilabel_confusion_matrix

if __name__ == "__main__":
    ds = CSVReader(
        [CSVFile('src/resources/iris.csv', delimiter=',', header=True, to_numeric=True, one_hot_classes=3)])
    files = ds.read_csv()
    # ts_file = files[0]
    # ts = TimeSeries(ts_file, 5, 3, NormalizationMethod.min_max, repeat=100)
    # x, y = ts.generate()

    file = files[0]
    x = file.data
    y = file.labels

    model = tf_model(input_dims=(4, ), output_dims=3, mode='classification')
    model.fit(x, y, epochs=500, verbose=0)

    x_test = x.copy()
    y_pred = model.predict(x_test)
    row_maxes = y_pred.max(axis=1).reshape(-1, 1)
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    print(multilabel_confusion_matrix(b, y))
