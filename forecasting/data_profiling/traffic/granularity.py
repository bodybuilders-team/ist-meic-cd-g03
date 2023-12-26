import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"
traffic_file_tag: str = "traffic_Total"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = traffic_data[target]

grans: list[str] = ["H", "D", "W"]  # , "M", "Q"]
# Does not make sense to aggregate by month or quarter because we only have 1 month of data

fig: Figure
axs: list[Axes]
fig, axs = plt.subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
fig.suptitle(f"{traffic_file_tag} {target} aggregation study")

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
plt.savefig(f"images/granularity/{traffic_file_tag}_aggregation_study.png")
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
    plt.savefig(f"images/granularity/{traffic_file_tag}_aggregation_study_{grans[i]}.png")
    plt.show()
    plt.clf()