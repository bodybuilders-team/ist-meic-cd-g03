import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
Scaling transformations are useful to reduce all numeric variables to a same range, in order to guarantee that variables
with larger scales do not assume more importance.
"""

credit_score_filename: str = "../../data/credit_score/processed_data/class_credit_score_truncate_outliers.csv"  # After truncating outliers
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(
    credit_score_filename)  # TODO , index_col="ID" - ID column was removed after encoding and I don't know why
target: str = "Credit_Score"
print(f"Dataset nr records={credit_score_data.shape[0]}", f"nr variables={credit_score_data.shape[1]}")

# ------------------
# Approach 1: Standard Scaler
# ------------------

vars: list[str] = credit_score_data.columns.to_list()
target_data: Series = credit_score_data.pop(target)

transform: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(credit_score_data)
df_zscore = DataFrame(transform.transform(credit_score_data), index=credit_score_data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_scaled_zscore.csv")

# ------------------
# Approach 2: MinMax Scaler
# ------------------

transform: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(credit_score_data)
df_minmax = DataFrame(transform.transform(credit_score_data), index=credit_score_data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_scaled_minmax.csv")

# To see the results:

fig, axs = plt.subplots(1, 3, figsize=(50, 10), squeeze=False)

axs[0, 0].set_title("Original data")
credit_score_data.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(labelrotation=90)

axs[0, 1].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 1].tick_params(labelrotation=90)

axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
axs[0, 2].tick_params(labelrotation=90)

plt.tight_layout()
plt.savefig(f"images/{credit_score_file_tag}_scaling.png")
plt.show()
plt.clf()
