import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series

from utils.dslabs_functions import ts_aggregation_by, plot_line_chart, HEIGHT

covid_filename: str = "../../data/covid/processed_data/forecast_covid_scaled.csv"
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = covid_data[target]

grans: dict = {"weekly": "W", "monthly": "M", "quarterly": "Q"}

for gran, freq in grans.items():
    covid_data_gran: DataFrame = ts_aggregation_by(series, gran_level=freq, agg_func="sum")
    plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    plot_line_chart(
        covid_data_gran.index.to_list(),
        covid_data_gran.to_list(),
        xlabel=covid_data_gran.index.name,
        ylabel=target,
        title=f"{covid_file_tag} {target} {gran} aggregation",
    )
    plt.tight_layout()
    plt.savefig(f"images/{covid_file_tag}_{gran}_aggregation.png")
    plt.show()
    plt.clf()

    # Save aggregated data
    covid_data_gran.to_csv(f"../../data/covid/processed_data/forecast_covid_{gran}_aggregated.csv")

# TODO: Choose the best aggregation: weekly, monthly or quarterly?
