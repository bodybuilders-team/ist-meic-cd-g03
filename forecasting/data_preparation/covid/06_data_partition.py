from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import series_train_test_split

covid_filename: str = "../../data/covid/forecast_covid.csv"  # TODO: Get data from differentiated data
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]

# ------------------
# Splitting the data into train and test series
# ------------------

train, test = series_train_test_split(series)

train.to_csv(f"../../data/covid/processed_data/{covid_file_tag}_train.csv", index=True)
test.to_csv(f"../../data/covid/processed_data/{covid_file_tag}_test.csv", index=True)
