import matplotlib.pyplot as plt
from numpy import array, ndarray
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split

from utils.dslabs_functions import plot_multibar_chart

# TODO: after finishing scaling, use the data from the selected scaling approach
pos_covid_filename: str = "../../data/pos_covid/processed_data/class_pos_covid_scaled_minmax.csv"  # After scaling
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename)
target: str = "CovidPos"

labels: list = list(pos_covid_data[target].unique())
labels.sort()
print(f"Labels={labels}")

positive: int = 1
negative: int = 0
values: dict[str, list[int]] = {
    "Original": [
        len(pos_covid_data[pos_covid_data[target] == negative]),
        len(pos_covid_data[pos_covid_data[target] == positive]),
    ]
}

y: array = pos_covid_data.pop(target).to_list()
X: ndarray = pos_covid_data.values

# ------------------
# Splitting the data into train and test sets
# ------------------

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train: DataFrame = concat([DataFrame(trnX, columns=pos_covid_data.columns), DataFrame(trnY, columns=[target])], axis=1)
train.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_train.csv", index=False)

test: DataFrame = concat(
    [DataFrame(tstX, columns=pos_covid_data.columns), DataFrame(tstY, columns=[target])], axis=1
)
test.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_test.csv", index=False)

values["Train"] = [
    len(train[train[target] == negative]),
    len(train[train[target] == positive]),
]
values["Test"] = [
    len(test[test[target] == negative]),
    len(test[test[target] == positive]),
]

plt.figure(figsize=(6, 4))
plot_multibar_chart(labels, values, title="Data distribution per dataset")
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_data_distribution.png")
plt.show()
plt.clf()
