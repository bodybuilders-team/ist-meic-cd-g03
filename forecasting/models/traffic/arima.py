import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA

from utils.dslabs_functions import series_train_test_split, HEIGHT, arima_study, plot_forecasting_eval, \
    plot_forecasting_series

traffic_filename: str = "../../data/traffic/forecast_traffic_first_diff.csv"  # TODO: Get data from differentiated data (DONE?)
traffic_file_tag: str = "traffic"
index_col: str = "Timestamp"
target: str = "Total"
measure: str = "R2"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]
train, test = series_train_test_split(series, trn_pct=0.90)

predictor = ARIMA(train, order=(3, 1, 2))
model = predictor.fit()
print(model.summary())

model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_arima_diagnostics.png")
plt.show()
plt.clf()

best_model, best_params = arima_study(train, test, measure=measure)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_arima_{measure}_study.png")
plt.show()
plt.clf()

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"{traffic_file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_arima_{measure}_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{traffic_file_tag} - ARIMA ",
    xlabel=index_col,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_arima_{measure}_forecast.png")
plt.show()
plt.clf()
