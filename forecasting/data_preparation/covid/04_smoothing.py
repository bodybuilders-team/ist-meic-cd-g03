import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT

covid_filename: str = "../../data/covid/forecast_covid.csv"  # TODO: Get data from aggregated data
covid_file_tag: str = "covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]

sizes: list[int] = [25, 50, 75, 100]
fig: Figure
axs: list[Axes]
fig, axs = plt.subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
fig.suptitle(f"{covid_file_tag} {target} after smoothing")

for i in range(len(sizes)):
    ss_smooth: Series = series.rolling(window=sizes[i]).mean()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        ax=axs[i],
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"size={sizes[i]}",
    )
plt.tight_layout()
plt.savefig(f"images/{covid_file_tag}_{target}_after_smoothing.png")
plt.show()
plt.clf()

# Save smoothed data
smoothed_data = series.rolling(window=50).mean()  # TODO: Choose best size
smoothed_data.to_csv(f"../../data/covid/processed_data/{covid_file_tag}_{target}_smoothed.csv")
