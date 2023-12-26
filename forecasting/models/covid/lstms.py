import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from forecasting.models.DS_LSTM import DS_LSTM, prepare_dataset_for_lstm
from utils.dslabs_functions import series_train_test_split, lstm_study, plot_forecasting_eval, plot_forecasting_series

covid_filename: str = "../../data/covid/processed_data/forecast_covid_first_diff.csv"
covid_file_tag: str = "covid"
timecol: str = "date"
target: str = "deaths"
measure: str = "R2"
covid_data: DataFrame = read_csv(covid_filename, index_col=timecol, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
train, test = series_train_test_split(series, trn_pct=0.90)

model = DS_LSTM(train, input_size=1, hidden_size=50, num_layers=1)
loss = model.fit()
print(loss)

best_model, best_params = lstm_study(train, test, nr_episodes=3000, measure=measure)

params = best_params["params"]
best_length = params[0]
trnX, trnY = prepare_dataset_for_lstm(train, seq_length=best_length)
tstX, tstY = prepare_dataset_for_lstm(test, seq_length=best_length)

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plot_forecasting_eval(
    train[best_length:],
    test[best_length:],
    prd_trn,
    prd_tst,
    title=f"{covid_file_tag} - LSTM (length={best_length}, hidden={params[1]}, epochs={params[2]})",
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_lstms_{measure}_eval.png")
plt.show()
plt.clf()

pred_series: Series = Series(prd_tst.numpy().ravel(), index=test.index[best_length:])

plot_forecasting_series(
    train[best_length:],
    test[best_length:],
    pred_series,
    title=f"{covid_file_tag} - LSTMs ",
    xlabel=timecol,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_lstms_{measure}_forecast.png")
plt.show()
plt.clf()
