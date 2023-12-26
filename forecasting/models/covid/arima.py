import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA

from utils.dslabs_functions import series_train_test_split, HEIGHT, arima_study, plot_forecasting_eval, \
    plot_forecasting_series

covid_filename: str = "../../data/covid/forecast_covid_first_diff.csv"  # TODO: Get data from differentiated data (DONE?)
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
measure: str = "R2"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
train, test = series_train_test_split(covid_data)

predictor = ARIMA(train, order=(3, 1, 2))
model = predictor.fit()
print(model.summary())

model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_arima_{measure}_diagnostics.png")
plt.show()
plt.clf()

best_model, best_params = arima_study(train, test, measure=measure)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_arima_{measure}_study.png")
plt.show()
plt.clf()

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(
    train, test, prd_trn, prd_tst, title=f"{covid_file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_arima_{measure}_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{covid_file_tag} - ARIMA ",
    xlabel=index_col,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_arima_{measure}_forecast.png")
plt.show()
plt.clf()
