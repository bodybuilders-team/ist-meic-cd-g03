import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT

traffic_filename: str = "../../data/traffic/processed_data/forecast_traffic_hourly_aggregated.csv"
traffic_file_tag: str = "forecast_traffic"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

sizes: list[int] = [25, 50, 75, 100]

for i in range(len(sizes)):
    fig = plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    ss_smooth: Series = series.rolling(window=sizes[i]).mean().dropna()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"{traffic_file_tag} {target} smoothed with size {sizes[i]}",
    )
    plt.tight_layout()
    plt.savefig(f"images/{traffic_file_tag}_smoothed_size_{sizes[i]}.png")
    plt.show()
    plt.clf()

    # Save smoothed data
    ss_smooth.to_csv(f"../../data/traffic/processed_data/{traffic_file_tag}_smoothed_size_{sizes[i]}.csv")
