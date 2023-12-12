import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import array
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import set_chart_labels, HEIGHT, ts_aggregation_by, get_lagged_series, plot_multiline_chart, \
    autocorrelation_study

covid_filename: str = "../../data/covid/forecast_covid.csv"
covid_file_tag: str = "covid"
index: str = "date"
covid_data: DataFrame = read_csv(covid_filename, index_col=index, parse_dates=True, infer_datetime_format=True)
target: str = "deaths"

series: Series = covid_data[target]

ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)
ss_months: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
ss_quarters: Series = ts_aggregation_by(series, gran_level="Q", agg_func=sum)

# ------------------
# 5-Number Summary
# ------------------

fig: Figure
axs: array
fig, axs = plt.subplots(2, 4, figsize=(4 * HEIGHT, HEIGHT))
set_chart_labels(axs[0, 0], title="DAILY")
axs[0, 0].boxplot(series)

set_chart_labels(axs[0, 1], title="WEEKLY")
axs[0, 1].boxplot(ss_weeks)

set_chart_labels(axs[0, 2], title="MONTHLY")
axs[0, 2].boxplot(ss_months)

set_chart_labels(axs[0, 3], title="QUARTERLY")
axs[0, 3].boxplot(ss_quarters)

axs[1, 0].grid(False)
axs[1, 0].set_axis_off()
axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")

axs[1, 1].grid(False)
axs[1, 1].set_axis_off()
axs[1, 1].text(0.2, 0, str(ss_weeks.describe()), fontsize="small")

axs[1, 2].grid(False)
axs[1, 2].set_axis_off()
axs[1, 2].text(0.2, 0, str(ss_months.describe()), fontsize="small")

axs[1, 3].grid(False)
axs[1, 3].set_axis_off()
axs[1, 3].text(0.2, 0, str(ss_quarters.describe()), fontsize="small")

plt.tight_layout()
plt.savefig(f"images/distribution/{covid_file_tag}_{target}_boxplot.png")
plt.show()
plt.clf()

# ------------------
# Variables Distribution
# ------------------

grans: list[Series] = [series, ss_weeks, ss_months, ss_quarters]
gran_names: list[str] = ["Daily", "Weekly", "Monthly", "Quarterly"]
fig: Figure
axs: array
fig, axs = plt.subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
fig.suptitle(f"{covid_file_tag} {target}")
for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values)
plt.tight_layout()
plt.savefig(f"images/distribution/{covid_file_tag}_{target}_distribution.png")
plt.show()
plt.clf()

# ------------------
# Autocorrelation - Lag Plot
# ------------------

plt.figure(figsize=(3 * HEIGHT, HEIGHT))
lags = get_lagged_series(series, 20, 10)
plot_multiline_chart(series.index.to_list(), lags, xlabel=index, ylabel=target)
plt.tight_layout()
plt.savefig(f"images/distribution/{covid_file_tag}_{target}_lag_plot.png")
plt.show()
plt.clf()

# ------------------
# Autocorrelation - Correlogram
# ------------------

autocorrelation_study(series, 10, 1)
plt.tight_layout()
plt.savefig(f"images/distribution/{covid_file_tag}_{target}_correlogram.png")
plt.show()
plt.clf()
