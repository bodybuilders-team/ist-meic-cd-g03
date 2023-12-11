import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from forecasting.models.SimpleAvgRegressor import SimpleAvgRegressor
from utils.dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from differentiated data
traffic_file_tag: str = "traffic"
timecol: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=timecol, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

train, test = series_train_test_split(series)

# Fit the model and predict

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{traffic_file_tag} - Simple Average")
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_simpleAvg_eval.png")
plt.show()
plt.clf()

# Plot the forecast

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{traffic_file_tag} - Simple Average",
    xlabel=timecol,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_simpleAvg_forecast.png")
plt.show()
plt.clf()