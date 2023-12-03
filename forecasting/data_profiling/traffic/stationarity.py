import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_components

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"
traffic_file_tag: str = "traffic"
traffic_data: DataFrame = read_csv(traffic_filename, index_col="Timestamp", parse_dates=True, infer_datetime_format=True)
target: str = "Total"
series: Series = traffic_data[target]
# each record is 15 minutes apart
series = series.resample("H").sum() # TODO: This is sus but I was getting an error when I tried to run plot_components without this

# Plot the components of the time series
plot_components(series, title=f"{traffic_file_tag} {target} components", x_label=series.index.name, y_label=target)
plt.tight_layout()
plt.savefig(f"images/stationarity/{traffic_file_tag}_{target}_components.png")
plt.show()
plt.clf()


# Stationarity study
# TODO: implement this - I dont know how to do it yet