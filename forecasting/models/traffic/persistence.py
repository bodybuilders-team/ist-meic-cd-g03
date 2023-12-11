import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from forecasting.models.PersistenceOptimistRegressor import PersistenceOptimistRegressor
from utils.dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from differentiated data
traffic_file_tag: str = "traffic"
timecol: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=timecol, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

train, test = series_train_test_split(series, trn_pct=0.90)

# ------

fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{traffic_file_tag} - Persistence Optimist")
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_persistence_optim_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{traffic_file_tag} - Persistence Optimist",
    xlabel=timecol,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_persistence_optim_forecast.png")
plt.show()
plt.clf()
