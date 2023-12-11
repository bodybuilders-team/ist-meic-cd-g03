import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_components, HEIGHT, plot_line_chart, eval_stationarity

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"
traffic_file_tag: str = "traffic"
target: str = "Total"
index: str = "Timestamp"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index, parse_dates=True, infer_datetime_format=True)

series: Series = traffic_data[target]

# ------------------
# Plot components
# ------------------
plot_components(series, title=f"{traffic_file_tag} {target} components", x_label=series.index.name, y_label=target)
plt.tight_layout()
plt.savefig(f"images/stationarity/{traffic_file_tag}_{target}_components.png")
plt.show()
plt.clf()

# ------------------
# Stationarity study
# ------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{traffic_file_tag} stationary study",  # TODO: not working because the teacher's function is not updated
    name="original"
)
n: int = len(series)
plot(series.index, [series.mean()] * n, "r-", label="mean")
plt.legend()
plt.tight_layout()
plt.savefig(f"images/stationarity/{traffic_file_tag}_{target}_stationary.png")
plt.show()
plt.clf()

BINS = 10
mean_line: list[float] = []

for i in range(BINS):
    segment: Series = series[i * n // BINS: (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{traffic_file_tag} stationary study",
    name="original",
    show_stdev=True
)
n: int = len(series)
plot(series.index, mean_line, "r-", label="mean")
plt.legend()
plt.tight_layout()
plt.savefig(f"images/stationarity/{traffic_file_tag}_{target}_stationary.png")
plt.show()
plt.clf()

# ------------------
# Augmented Dickey-Fuller test
# ------------------

print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary")
