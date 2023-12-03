import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_components

covid_filename: str = "../../data/covid/forecast_covid.csv"
covid_file_tag: str = "covid"
covid_data: DataFrame = read_csv(covid_filename, index_col="date", parse_dates=True, infer_datetime_format=True)
target: str = "deaths"

series: Series = covid_data[target]

# Plot the components of the time series
plot_components(series, title=f"{covid_file_tag} {target} components", x_label=series.index.name, y_label=target)
plt.tight_layout()
plt.savefig(f"images/stationarity/{covid_file_tag}_{target}_components.png")
plt.show()
plt.clf()


# Stationarity study
# TODO: implement this - I dont know how to do it yet