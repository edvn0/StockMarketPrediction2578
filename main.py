import numpy as np
from src.data_prediction import tf_model
from src.normalization import NormalizationMethod
from src.data_input import CSVReader, CSVFile
from src.data_analysis import DataAnalysis
from src.time_series import TimeSeries
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ds = CSVReader(data_indices=[1, 3, 4],
                   label_index=2, filenames=None, dir='src/stocks')
    files = ds.read_csv()
    ts_file = files[0]
    ts = TimeSeries(ts_file, 3, 1, NormalizationMethod.min_max)
    X, Y = ts.generate(split=True)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.15)
    model = tf_model(input_dims=(3, 3), output_dims=1, mode='regression')
    model.fit(x_train, y_train, batch_size=25, epochs=10, verbose=2)

    y_pred = model.predict(x_test)
    row_maxes = y_pred.max(axis=1).reshape(-1, 1)
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    print(multilabel_confusion_matrix(b, y))
