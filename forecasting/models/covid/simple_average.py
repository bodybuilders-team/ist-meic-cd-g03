import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from forecasting.models.SimpleAvgRegressor import SimpleAvgRegressor
from utils.dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

covid_filename: str = "../../data/covid/processed_data/forecast_covid_first_diff.csv"
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
train, test = series_train_test_split(covid_data)

# Fit the model and predict

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{covid_file_tag} - Simple Average")
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_simpleAvg_eval.png")
plt.show()
plt.clf()

# Plot the forecast

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{covid_file_tag} - Simple Average",
    xlabel=index_col,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_simpleAvg_forecast.png")
plt.show()
plt.clf()
