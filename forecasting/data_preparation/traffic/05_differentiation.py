from pandas import DataFrame, read_csv

from utils.dslabs_functions import plot_ts_multivariate_chart
import matplotlib.pyplot as plt

traffic_filename: str = "../../data/traffic/forecast_traffic.csv" # TODO: Get data from smoothing
traffic_file_tag: str = "traffic"
traffic_data: DataFrame = read_csv(traffic_filename, index_col="Timestamp", parse_dates=True, infer_datetime_format=True)
target: str = "Total"

# Remove Week variable, because it is not relevant for the following analysis
traffic_data = traffic_data.drop(columns=["Day of the week"], inplace=False)


plot_ts_multivariate_chart(traffic_data, title=f"{traffic_file_tag} {target}")
plt.tight_layout()
plt.show()
plt.clf()

# ----------------------------
# First differentiation
# ----------------------------

diff_df: DataFrame = traffic_data.diff()

plot_ts_multivariate_chart(diff_df, title=f"{traffic_file_tag} {target} - after first differentiation")
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_{target}_first_diff.png")
plt.show()
plt.clf()

diff_df.to_csv(f"../../data/traffic/processed_data/forecast_traffic_{target}_first_diff.csv")

# ----------------------------
# Second differentiation
# ----------------------------
diff_df: DataFrame = diff_df.diff()

plot_ts_multivariate_chart(diff_df, title=f"{traffic_file_tag} {target} - after second differentiation")
plt.tight_layout()
plt.savefig(f"images/{traffic_file_tag}_{target}_second_diff.png")
plt.show()
plt.clf()

diff_df.to_csv(f"../../data/traffic/processed_data/forecast_traffic_{target}_second_diff.csv")
