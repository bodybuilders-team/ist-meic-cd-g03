import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import array
from pandas import DataFrame, read_csv, Series

from utils.dslabs_functions import set_chart_labels, HEIGHT, ts_aggregation_by

traffic_filename: str = "../../data/traffic/forecast_traffic.csv"
traffic_file_tag: str = "traffic"
traffic_data: DataFrame = read_csv(traffic_filename, index_col="Timestamp", parse_dates=True, infer_datetime_format=True)
target: str = "Total"

series: Series = traffic_data[target]

ss_days: Series = ts_aggregation_by(series, gran_level="D", agg_func=sum)
ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)
# Does not make sense to aggregate by month or quarter because we only have 1 month of data
# ss_months: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
# ss_quarters: Series = ts_aggregation_by(series, gran_level="Q", agg_func=sum)

# ------------------
# 5-Number Summary
# ------------------

fig: Figure
axs: array
fig, axs = plt.subplots(2, 3, figsize=(4 * HEIGHT, HEIGHT))
set_chart_labels(axs[0, 0], title="HOURLY")
axs[0, 0].boxplot(series)

set_chart_labels(axs[0, 1], title="DAILY")
axs[0, 1].boxplot(ss_days)

set_chart_labels(axs[0, 2], title="WEEKLY")
axs[0, 2].boxplot(ss_weeks)

# Does not make sense to aggregate by month or quarter because we only have 1 month of data
# set_chart_labels(axs[0, 3], title="MONTHLY")
# axs[0, 3].boxplot(ss_months)
#
# set_chart_labels(axs[0, 4], title="QUARTERLY")
# axs[0, 4].boxplot(ss_quarters)

axs[1, 0].grid(False)
axs[1, 0].set_axis_off()
axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")

axs[1, 1].grid(False)
axs[1, 1].set_axis_off()
axs[1, 1].text(0.2, 0, str(ss_days.describe()), fontsize="small")

axs[1, 2].grid(False)
axs[1, 2].set_axis_off()
axs[1, 2].text(0.2, 0, str(ss_weeks.describe()), fontsize="small")

# Does not make sense to aggregate by month or quarter because we only have 1 month of data
# axs[1, 3].grid(False)
# axs[1, 3].set_axis_off()
# axs[1, 3].text(0.2, 0, str(ss_months.describe()), fontsize="small")
#
# axs[1, 4].grid(False)
# axs[1, 4].set_axis_off()
# axs[1, 4].text(0.2, 0, str(ss_quarters.describe()), fontsize="small")

plt.tight_layout()
plt.savefig(f"images/distribution/{traffic_file_tag}_{target}_boxplot.png")
plt.show()
plt.clf()

# ------------------
# Variables Distribution
# ------------------

grans: list[Series] = [series, ss_days, ss_weeks]  # , ss_months, ss_quarters]
gran_names: list[str] = ["Hourly", "Daily", "Weekly"]  # , "Monthly", "Quarterly"]
fig: Figure
axs: array
fig, axs = plt.subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
fig.suptitle(f"{traffic_file_tag} {target}")
for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values)
plt.tight_layout()
plt.savefig(f"images/distribution/{traffic_file_tag}_{target}_distribution.png")
plt.show()
plt.clf()
