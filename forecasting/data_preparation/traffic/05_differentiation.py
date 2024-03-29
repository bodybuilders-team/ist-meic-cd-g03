import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import HEIGHT, plot_line_chart

traffic_filename: str = "../../data/traffic/processed_data/forecast_traffic_smoothed_size_50.csv"
traffic_file_tag: str = "forecast_traffic"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = traffic_data[target]

# ----------------------------
# No differentiation
# ----------------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{traffic_file_tag} {target} - no differentiation",
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_no_diff.png")
plt.show()
plt.clf()

# ----------------------------
# First differentiation
# ----------------------------

ss_diff: Series = series.diff().dropna()

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title=f"{traffic_file_tag} {target} - after first differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_first_diff.png")
plt.show()
plt.clf()

ss_diff.to_csv(f"../../data/traffic/processed_data/forecast_traffic_first_diff.csv")

# ----------------------------
# Second differentiation
# ----------------------------
ss_diff: Series = ss_diff.diff().dropna()

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title=f"{traffic_file_tag} {target} - after second differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_second_diff.png")
plt.show()
plt.clf()

ss_diff.to_csv(f"../../data/traffic/processed_data/forecast_traffic_second_diff.csv")
