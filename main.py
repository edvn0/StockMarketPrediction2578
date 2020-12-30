import numpy as np
from src.data_prediction import tf_model
from src.normalization import NormalizationMethod, Normalizer
from src.data_input import CSVReader, CSVFile
from src.data_analysis import DataAnalysis
from src.time_series import TimeSeries
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy.signal import correlate

if __name__ == "__main__":
    ds = CSVReader(feature_index=2, info_indices=[
                   0], filenames=None, dir='src/resources/stocks/fixed')
    datasets = ds.read_csv()

    for dataset in datasets:
        normer = Normalizer(dataset, NormalizationMethod.min_max)
        ts = TimeSeries(normer, window_size=6, label_size=1)

        X, Y = ts.generate(split=True)
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.25, shuffle=False)
        model = tf_model(input_dims=(1, 3), output_dims=1)
        model.fit(x_train, y_train, epochs=1, verbose=1)

        x_dates = range(len(x_test))
        predictions = [y[0] for y in model.predict(x_test)]
        rescaled_preds = normer.denormalize(predictions)
        real = [y[0] for y in y_test]
        renormed = correlate(
            real, rescaled_preds, method='fft')
        correl = [y[0]
                  for y in renormed]

        with open(f'prediction_{dataset.identifier()}', 'w') as f:
            f.write("date_index,pred,real,correl\n")
            for val in zip(x_dates, predictions, real, correl):
                actual = (val[0], val[1][0], val[2][0], val[3])
                f.write(','.join(str(e) for e in actual) + "\n")
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(x_dates, predictions, label="Predicted")
        ax2.plot(x_dates, real, label="Real")
        fig.savefig(f"predicted_stocks_{dataset.identifier()}.png")
