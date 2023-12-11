import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from utils.dslabs_functions import series_train_test_split, HEIGHT, rolling_mean_study, plot_forecasting_eval, \
    plot_forecasting_series

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from differentiated data
traffic_file_tag: str = "traffic"
timecol: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=timecol, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

train, test = series_train_test_split(series, trn_pct=0.90)

measure: str = "R2"

fig = plt.figure(figsize=(HEIGHT, HEIGHT))
best_model, best_params = rolling_mean_study(train, test)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_rollingmean_{measure}_study.png")
plt.show()
plt.clf()

params = best_params["params"]
prd_trn: Series = best_model.predict(train)
prd_tst: Series = best_model.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{traffic_file_tag} - Rolling Mean (win={params[0]})")
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{traffic_file_tag} - Rolling Mean (win={params[0]})",
    xlabel=timecol,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_rollingmean_{measure}_forecast.png")
plt.show()
plt.clf()
