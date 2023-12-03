import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import plot_ts_multivariate_chart, ts_aggregation_by

covid_filename: str = "../../data/covid/forecast_covid.csv"  # TODO: Get data from aggregated data
covid_file_tag: str = "covid"
covid_data: DataFrame = read_csv(covid_filename, index_col="date", parse_dates=True, infer_datetime_format=True)
target: str = "deaths"

# Remove Week variable, because it is not relevant for the following analysis
covid_data = covid_data.drop(columns=["week"], inplace=False)

plot_ts_multivariate_chart(covid_data, title=f"{covid_file_tag} {target}")
plt.tight_layout()
plt.show()
plt.clf()

grans: dict = {"weekly": "W", "monthly": "M", "quarterly": "Q"}

for gran, freq in grans.items():
    covid_data_gran: DataFrame = ts_aggregation_by(covid_data, gran_level=freq, agg_func="sum")
    plot_ts_multivariate_chart(covid_data_gran, title=f"{covid_file_tag} {target} {gran} aggregation")
    plt.tight_layout()
    plt.savefig(f"images/{covid_file_tag}_{target}_{gran}_aggregation.png")
    plt.show()
    plt.clf()

    # Save aggregated data
    covid_data_gran.to_csv(f"../../data/covid/processed_data/forecast_covid_{target}_{gran}_aggregated.csv")

# TODO: Choose the best aggregation: weekly, monthly or quarterly?
