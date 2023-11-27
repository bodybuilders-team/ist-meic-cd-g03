import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
Scaling transformations are useful to reduce all numeric variables to a same range, in order to guarantee that variables
with larger scales do not assume more importance.
"""

pos_covid_filename: str = "../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv"  # After truncating outliers
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename)
target: str = "CovidPos"
print(f"Dataset nr records={pos_covid_data.shape[0]}", f"nr variables={pos_covid_data.shape[1]}")

# ------------------
# Approach 1: Standard Scaler
# ------------------

vars: list[str] = pos_covid_data.columns.to_list()
target_data: Series = pos_covid_data.pop(target)
print(pos_covid_data)

transform: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(pos_covid_data)
df_zscore = DataFrame(transform.transform(pos_covid_data), index=pos_covid_data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_scaled_zscore.csv", index=False)

# ------------------
# Approach 2: MinMax Scaler
# ------------------

transform: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(pos_covid_data)
df_minmax = DataFrame(transform.transform(pos_covid_data), index=pos_covid_data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_scaled_minmax.csv", index=False)

# To see the results:

fig, axs = plt.subplots(1, 3, figsize=(50, 10), squeeze=False)

axs[0, 0].set_title("Original data")
pos_covid_data.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(labelrotation=90)

axs[0, 1].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 1].tick_params(labelrotation=90)

axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
axs[0, 2].tick_params(labelrotation=90)

plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_scaling.png")
plt.show()
plt.clf()
