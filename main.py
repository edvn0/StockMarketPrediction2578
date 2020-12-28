import numpy as np
from src.data_prediction import tf_model
from src.normalization import NormalizationMethod
from src.data_input import CSVReader, CSVFile
from src.data_analysis import DataAnalysis
from src.time_series import TimeSeries
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy.signal import correlate

if __name__ == "__main__":
    ds = CSVReader(data_indices=[4],
                   label_index=2, info_indices=[0], filenames=None, dir='src/resources/stocks/fixed')
    files = ds.read_csv()

    for file in files:
        ts_file = file
        ts = TimeSeries(ts_file, 1, 1, NormalizationMethod.min_max)
        X, Y = ts.generate(split=True)
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.25)
        model = tf_model(input_dims=(1, 1), output_dims=1)
        model.fit(x_train, y_train, epochs=100, verbose=2)

        x_dates = range(len(x_test))
        predictions = [y[0] for y in model.predict(x_test)]
        real = [y[0] for y in y_test]
        correl = [y[0]
                  for y in correlate(real, predictions, method='fft') / 500]

        with open(f'prediction_{file.identifier()}', 'w') as f:
            f.write("date_index,pred,real,correl\n")
            for val in zip(x_dates, predictions, real, correl):
                actual = (val[0], val[1][0], val[2][0], val[3])
                f.write(','.join(str(e) for e in actual) + "\n")

        plt.plot(x_dates, predictions, label="Predicted")
        plt.plot(x_dates, real, label="Real")
        plt.savefig(f"predicted_stocks_{file.identifier()}.png")
        plt.reset
