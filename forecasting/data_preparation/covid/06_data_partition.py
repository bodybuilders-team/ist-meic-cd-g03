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

covid_filename: str = "../../data/covid/forecast_covid.csv"  # TODO: Get data from differentiated data
covid_file_tag: str = "covid"
covid_data: DataFrame = read_csv(covid_filename, index_col="date", parse_dates=True, infer_datetime_format=True)
target: str = "deaths"

series: Series = covid_data[target]

# ------------------
# Splitting the data into train and test series
# ------------------

train, test = series_train_test_split(series)

train.to_csv(f"../../data/covid/processed_data/{covid_file_tag}_train.csv", index=False)
test.to_csv(f"../../data/covid/processed_data/{covid_file_tag}_test.csv", index=False)
