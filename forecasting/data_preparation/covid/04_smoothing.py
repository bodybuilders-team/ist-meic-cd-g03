import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import plot_line_chart, HEIGHT

covid_filename: str = "../../data/covid/processed_data/forecast_covid_weekly_aggregated.csv"
covid_file_tag: str = "forecast_covid"
index_col: str = "date"
target: str = "deaths"
covid_data: DataFrame = read_csv(covid_filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)

series: Series = covid_data[target]

sizes: list[int] = [25, 50, 75, 100]
fig: Figure
axs: list[Axes]

for i in range(len(sizes)):
    fig = plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    ss_smooth: Series = series.rolling(window=sizes[i]).mean().dropna()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"{covid_file_tag} {target} smoothed with size {sizes[i]}",
    )
    plt.tight_layout()
    plt.savefig(f"images/{covid_file_tag}_smoothed_size_{sizes[i]}.png")
    plt.show()
    plt.clf()

    # Save smoothed data
    ss_smooth.to_csv(f"../../data/covid/processed_data/{covid_file_tag}_smoothed_size_{sizes[i]}.csv")
