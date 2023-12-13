import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from aggregated data
traffic_file_tag: str = "traffic"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

sizes: list[int] = [25, 50, 75, 100]
fig: Figure
axs: list[Axes]
fig, axs = plt.subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
fig.suptitle(f"{traffic_file_tag} {target} after smoothing")

for i in range(len(sizes)):
    ss_smooth: Series = series.rolling(window=sizes[i]).mean()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        ax=axs[i],
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"size={sizes[i]}",
    )
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_after_smoothing.png")
plt.show()
plt.clf()

# Save smoothed data
smoothed_data = series.rolling(window=50).mean()  # TODO: Choose best size
smoothed_data.to_csv(f"../../data/traffic/processed_data/{traffic_file_tag}_smoothed.csv")
