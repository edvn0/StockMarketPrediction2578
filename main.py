from src.data_input import CSVReader, CSVFile
from src.data_analysis import DataAnalysis
from src.time_series import TimeSeries

if __name__ == "__main__":
    ds = CSVReader(
        [CSVFile('src/resources/time_series_test.csv', delimiter=',', header=True, to_numeric=True)])
    files = ds.read_csv()
    ts_file = files[0]
    ts = TimeSeries(ts_file, 5, 3)
    created = ts.generate()

    print(created)
