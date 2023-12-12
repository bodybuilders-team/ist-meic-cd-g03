import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from utils.dslabs_functions import ts_aggregation_by, HEIGHT, plot_line_chart

traffic_filename: str = "../../data/traffic/processed_data/forecast_traffic_Total_scaled.csv"
traffic_file_tag: str = "traffic"
index_col: str = "Timestamp"
target: str = "Total"
traffic_data: DataFrame = read_csv(traffic_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = traffic_data[target]

grans: dict = {"hourly": "H", "daily": "D", "weekly": "W"}

for gran, freq in grans.items():
    traffic_data_gran: DataFrame = ts_aggregation_by(series, gran_level=freq, agg_func="sum")
    plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    plot_line_chart(
        traffic_data_gran.index.to_list(),
        traffic_data_gran.to_list(),
        xlabel=traffic_data_gran.index.name,
        ylabel=target,
        title=f"{traffic_file_tag} {target} {gran} aggregation",
    )
    plt.tight_layout()
    plt.savefig(f"images/{traffic_file_tag}_{target}_{gran}_aggregation.png")
    plt.show()
    plt.clf()

    # Save aggregated data
    traffic_data_gran.to_csv(f"../../data/traffic/processed_data/forecast_traffic_{target}_{gran}_aggregated.csv")

# TODO: Choose the best aggregation: hourly, daily or weekly?
