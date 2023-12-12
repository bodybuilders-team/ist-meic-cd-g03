import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT, scale_all_dataframe

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"
traffic_file_tag: str = "traffic"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = traffic_data[target]

# Before scaling

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{traffic_file_tag} {target} before scaling",
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_{target}_before_scaling.png")
plt.show()
plt.clf()

# After scaling

df: DataFrame = scale_all_dataframe(traffic_data)

ss: Series = df[target]
plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    ss.index.to_list(),
    ss.to_list(),
    xlabel=ss.index.name,
    ylabel=target,
    title=f"{traffic_file_tag} {target} after scaling",
)
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_{target}_after_scaling.png")
plt.show()
plt.clf()

# Save scaled data
df.to_csv(f"../../data/traffic/processed_data/forecast_traffic_{target}_scaled.csv")
