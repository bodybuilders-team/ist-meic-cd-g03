import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from forecasting.models.PersistenceOptimistRegressor import PersistenceOptimistRegressor
from forecasting.models.PersistenceRealistRegressor import PersistenceRealistRegressor
from utils.dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

covid_filename: str = "../../data/covid/forecast_covid.csv"  # TODO: Get data from differentiated data
covid_file_tag: str = "covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
train, test = series_train_test_split(covid_data)

# ------------------
# Persistence Optimist Regressor
# ------------------

fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{covid_file_tag} - Persistence Optimist")
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_persistence_optim_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{covid_file_tag} - Persistence Optimist",
    xlabel=index_col,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_persistence_optim_forecast.png")
plt.show()
plt.clf()

# ------------------
# Persistence Realist Regressor
# ------------------

fr_mod = PersistenceRealistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{covid_file_tag} - Persistence Realist")
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_persistence_real_eval.png")
plt.show()
plt.clf()

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{covid_file_tag} - Persistence Realist",
    xlabel=index_col,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_persistence_real_forecast.png")
plt.show()
plt.clf()
