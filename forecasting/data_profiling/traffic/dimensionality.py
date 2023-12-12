import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"
traffic_file_tag: str = "traffic"
traffic_data: DataFrame = read_csv(traffic_filename, index_col="Timestamp", parse_dates=True,
                                   infer_datetime_format=True)
target: str = "Total"

series: Series = traffic_data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

# Plot the data at the most atomic granularity - daily (not weekly - that will be studied in the granularity section)

# ------------------
# Daily Total Traffic
# ------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{traffic_file_tag} daily {target}",
)
plt.tight_layout()
plt.savefig(f"images/dimensionality/{traffic_file_tag}_daily_{target}.png")
plt.show()
plt.clf()

# ------------------
# Multivariate Time Series - All Variables
# ------------------

# Remove day of the week variable, because it is not relevant for the following analysis
traffic_data = traffic_data.drop(columns=["Day of the week"], inplace=False)

plot_ts_multivariate_chart(traffic_data, title=f"{traffic_file_tag} {target}")
plt.tight_layout()
plt.savefig(f"images/dimensionality/{traffic_file_tag}_{target}.png")
plt.show()
