from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import series_train_test_split

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from differentiated data
traffic_file_tag: str = "traffic"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

# ------------------
# Splitting the data into train and test series
# ------------------

train, test = series_train_test_split(traffic_data)

train.to_csv(f"../../data/traffic/processed_data/{traffic_file_tag}_train.csv", index=True)
test.to_csv(f"../../data/traffic/processed_data/{traffic_file_tag}_test.csv", index=True)
