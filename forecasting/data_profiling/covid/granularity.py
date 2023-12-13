import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by

covid_filename: str = "../../data/covid/forecast_covid.csv"
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = covid_data[target]

# All in the same plot

grans: list[str] = ["W", "M", "Q"]
fig: Figure
axs: list[Axes]
fig, axs = plt.subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
fig.suptitle(f"{covid_file_tag} {target} aggregation study")

for i in range(len(grans)):
    ss: Series = ts_aggregation_by(series, grans[i])
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        ax=axs[i],
        xlabel=f"{ss.index.name} ({grans[i]})",
        ylabel=target,
        title=f"granularity={grans[i]}",
    )
plt.tight_layout()
plt.savefig(f"images/granularity/{covid_file_tag}_aggregation_study.png")
plt.show()
plt.clf()

# Different plots

for i in range(len(grans)):
    fig = plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    ss: Series = ts_aggregation_by(series, grans[i])
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        xlabel=f"{ss.index.name} ({grans[i]})",
        ylabel=target,
        title=f"granularity={grans[i]}",
    )
    plt.tight_layout()
    plt.savefig(f"images/granularity/{covid_file_tag}_aggregation_study_{grans[i]}.png")
    plt.show()
    plt.clf()
