import matplotlib.pyplot as plt
from numpy import arange
from pandas import read_csv, DataFrame, Series
from sklearn.linear_model import LinearRegression

from utils.dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

covid_file_tag: str = "covid"
index_col: str = "date"
target: str = "deaths"

run_aggregation_study = True
run_smoothing_study = True

"""
------------------
Aggregation
------------------

% Approach 1: Weekly Aggregation
% Approach 2: Monthly Aggregation
% Approach 3: Quarterly Aggregation
"""

if run_aggregation_study:
    # TODO: Add aggregation study
    pass

if run_smoothing_study:
    # TODO: Add smoothing study
    pass

# Example of study
covid_filename: str = "../../data/covid/forecast_covid.csv"
covid_file_tag: str = "covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
train, test = series_train_test_split(covid_data)

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(covid_data)).reshape(-1, 1)
tstY = test.to_numpy()

model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX).reshape(-1), index=train.index)
prd_tst: Series = Series(model.predict(tstX).reshape(-1), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{covid_file_tag} - Linear Regression")
plt.show()
# savefig(f"images/{covid_file_tag}_linear_regression_eval.png")


plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{covid_file_tag} - Linear Regression Forecast",
    xlabel=index_col,
    ylabel=target,
)
plt.show()
# savefig(f"images/{covid_file_tag}_linear_regression_forecast.png")
