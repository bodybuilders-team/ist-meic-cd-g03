import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from utils.dslabs_functions import series_train_test_split, HEIGHT, rolling_mean_study, plot_forecasting_eval, \
    plot_forecasting_series

covid_filename: str = "../../data/covid/forecast_covid_first_diff.csv"  # TODO: Get data from differentiated data (DONE?)
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
train, test = series_train_test_split(series)

measure: str = "R2"

fig = plt.figure(figsize=(HEIGHT, HEIGHT))
best_model, best_params = rolling_mean_study(train, test)  # FIXME: This is not working
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_rollingmean_{measure}_study.png")
plt.show()
plt.clf()

params = best_params["params"]
prd_trn: Series = best_model.predict(train)
prd_tst: Series = best_model.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{covid_file_tag} - Rolling Mean (win={params[0]})")
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{covid_file_tag} - Rolling Mean (win={params[0]})",
    xlabel=index_col,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_rollingmean_{measure}_forecast.png")
plt.show()
plt.clf()
