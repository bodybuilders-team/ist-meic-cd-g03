import matplotlib.pyplot as plt
from numpy import array, ndarray
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split

from utils.dslabs_functions import plot_multibar_chart

# TODO: after finishing scaling, use the data from the selected scaling approach
credit_score_filename: str = "../../data/credit_score/processed_data/class_credit_score_scaled_minmax.csv"  # After scaling
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename)
target: str = "Credit_Score"

labels: list = list(credit_score_data[target].unique())
labels.sort()
print(f"Labels={labels}")

positive: int = 1
negative: int = 0
values: dict[str, list[int]] = {
    "Original": [
        len(credit_score_data[credit_score_data[target] == negative]),
        len(credit_score_data[credit_score_data[target] == positive]),
    ]
}

y: array = credit_score_data.pop(target).to_list()
X: ndarray = credit_score_data.values

# ------------------
# Splitting the data into train and test sets
# ------------------

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train: DataFrame = concat([DataFrame(trnX, columns=credit_score_data.columns), DataFrame(trnY, columns=[target])],
                          axis=1)
train.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_train.csv", index=False)

test: DataFrame = concat(
    [DataFrame(tstX, columns=credit_score_data.columns), DataFrame(tstY, columns=[target])], axis=1
)
test.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_test.csv", index=False)

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
plt.savefig(f"images/{credit_score_file_tag}_data_distribution.png")
plt.show()
plt.clf()
