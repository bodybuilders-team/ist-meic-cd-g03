import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT

covid_filename: str = "../../data/covid/forecast_covid.csv"
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

# Plot the data at the most atomic granularity - daily (not weekly - that will be studied in the granularity section)

# ------------------
# Weakly deaths
# ------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{covid_file_tag} weakly {target}",
)
plt.tight_layout()
plt.savefig(f"images/dimensionality/{covid_file_tag}_daily_{target}.png")
plt.show()
plt.clf()
