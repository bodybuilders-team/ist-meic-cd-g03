import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart

covid_filename: str = "../../data/covid/forecast_covid.csv"
covid_file_tag: str = "covid"
covid_data: DataFrame = read_csv(covid_filename, index_col="date", parse_dates=True, infer_datetime_format=True)
target: str = "deaths"

series: Series = covid_data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

# Plot the data at the most atomic granularity - daily (not weekly - that will be studied in the granularity section)

# ------------------
# Daily Deaths
# ------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{covid_file_tag} daily {target}",
)
plt.tight_layout()
plt.savefig(f"images/dimensionality/{covid_file_tag}_daily_{target}.png")
plt.show()
plt.clf()

# ------------------
# Multivariate Time Series - All Variables
# ------------------

# Remove Week variable, because it is not relevant for the following analysis
covid_data = covid_data.drop(columns=["week"], inplace=False)

plot_ts_multivariate_chart(covid_data, title=f"{covid_file_tag} {target}")
plt.tight_layout()
plt.savefig(f"images/dimensionality/{covid_file_tag}_{target}.png")
plt.show()
