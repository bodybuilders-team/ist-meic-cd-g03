from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import series_train_test_split

"""
Regarding data partition, remember that time series are temporal data, and so test data shall
always be posterior to any train data. Remember, that Persistence model predicts the
following value based on the last one known, so we can consider two scenarios: the best –
corresponding to the one-step horizon, and the rough one – when we use the last value of
the training set to predict all the future values. In this manner, this model provides us two
baselines for comparing all the other results.
"""

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from differentiated data
traffic_file_tag: str = "traffic"
traffic_data: DataFrame = read_csv(traffic_filename, index_col="Timestamp", parse_dates=True,
                                   infer_datetime_format=True)
target: str = "Total"

series: Series = traffic_data[target]

# ------------------
# Splitting the data into train and test series
# ------------------

train, test = series_train_test_split(series)

train.to_csv(f"../../data/traffic/processed_data/{traffic_file_tag}_train.csv", index=False)
test.to_csv(f"../../data/traffic/processed_data/{traffic_file_tag}_test.csv", index=False)
