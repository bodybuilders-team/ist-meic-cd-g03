from pandas import DataFrame, read_csv

from utils.dslabs_functions import plot_ts_multivariate_chart
import matplotlib.pyplot as plt

covid_filename: str = "../../data/covid/forecast_covid.csv" # TODO: Get data from smoothing
covid_file_tag: str = "covid"
covid_data: DataFrame = read_csv(covid_filename, index_col="date", parse_dates=True, infer_datetime_format=True)
target: str = "deaths"

# Remove Week variable, because it is not relevant for the following analysis
covid_data = covid_data.drop(columns=["week"], inplace=False)


plot_ts_multivariate_chart(covid_data, title=f"{covid_file_tag} {target}")
plt.tight_layout()
plt.show()
plt.clf()

# ----------------------------
# First differentiation
# ----------------------------

diff_df: DataFrame = covid_data.diff()

plot_ts_multivariate_chart(diff_df, title=f"{covid_file_tag} {target} - after first differentiation")
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_{target}_first_diff.png")
plt.show()
plt.clf()

diff_df.to_csv(f"../../data/covid/processed_data/forecast_covid_{target}_first_diff.csv")

# ----------------------------
# Second differentiation
# ----------------------------
diff_df: DataFrame = diff_df.diff()

plot_ts_multivariate_chart(diff_df, title=f"{covid_file_tag} {target} - after second differentiation")
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_{target}_second_diff.png")
plt.show()
plt.clf()

diff_df.to_csv(f"../../data/covid/processed_data/forecast_covid_{target}_second_diff.csv")
