import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import HEIGHT, plot_line_chart

covid_filename: str = "../../data/covid/forecast_covid.csv"  # TODO: Get data from smoothing
covid_file_tag: str = "covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
series: Series = covid_data[target]

# ----------------------------
# No differentiation
# ----------------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{covid_file_tag} {target} - no differentiation",
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_{target}_no_diff.png")
plt.show()
plt.clf()

# ----------------------------
# First differentiation
# ----------------------------

ss_diff: Series = series.diff()

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title=f"{covid_file_tag} {target} - after first differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_{target}_first_diff.png")
plt.show()
plt.clf()

ss_diff.to_csv(f"../../data/covid/processed_data/forecast_covid_{target}_first_diff.csv")

# ----------------------------
# Second differentiation
# ----------------------------
ss_diff: Series = ss_diff.diff()

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title=f"{covid_file_tag} {target} - after second differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_{target}_second_diff.png")
plt.show()
plt.clf()

ss_diff.to_csv(f"../../data/covid/processed_data/forecast_covid_{target}_second_diff.csv")
