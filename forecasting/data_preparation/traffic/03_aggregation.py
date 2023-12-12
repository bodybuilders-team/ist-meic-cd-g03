import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import plot_ts_multivariate_chart, ts_aggregation_by

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"  # TODO: Get data from aggregated data
traffic_file_tag: str = "traffic"
traffic_data: DataFrame = read_csv(traffic_filename, index_col="Timestamp", parse_dates=True,
                                   infer_datetime_format=True)
target: str = "Total"

# Remove Day of the weeK variable, because it is not relevant for the following analysis
traffic_data = traffic_data.drop(columns=["Day of the week"], inplace=False)

plot_ts_multivariate_chart(traffic_data, title=f"{traffic_file_tag} {target}")
plt.tight_layout()
plt.show()
plt.clf()

grans: dict = {"hourly": "H", "daily": "D", "weekly": "W"}

for gran, freq in grans.items():
    traffic_data_gran: DataFrame = ts_aggregation_by(traffic_data, gran_level=freq, agg_func="sum")
    plot_ts_multivariate_chart(traffic_data_gran, title=f"{traffic_file_tag} {target} {gran} aggregation")
    plt.tight_layout()
    plt.savefig(f"images/{traffic_file_tag}_{target}_{gran}_aggregation.png")
    plt.show()
    plt.clf()

    # Save aggregated data
    traffic_data_gran.to_csv(f"../../data/traffic/processed_data/forecast_traffic_{target}_{gran}_aggregated.csv")

# TODO: Choose the best aggregation: hourly, daily or weekly?
