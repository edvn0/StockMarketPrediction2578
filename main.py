from data_input import CSVReader, CSVFile
from data_analysis import DataAnalysis
from time_series import TimeSeries

if __name__ == "__main__":
    ds = CSVReader(
        [CSVFile('time_series_test.csv', delimiter=',', header=True, to_numeric=True)])
    files = ds.read_csv()
    ts_file = files[0]
    ts = TimeSeries(ts_file, 1, 1)
    created = ts.generate()
    print(created)
