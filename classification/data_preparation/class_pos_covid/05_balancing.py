import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from pandas import read_csv, DataFrame, Series, concat

from utils.dslabs_functions import plot_bar_chart

"""
A dataset is unbalanced if the number of samples in each class is not similar - this can bias the model.
"""

pos_covid_filename: str = "../../data/pos_covid/class_pos_covid.csv"
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
target: str = "CovidPos"

target_count: Series = pos_covid_data[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)
values: dict[str, list] = {
    "Original": [target_count[positive_class], target_count[negative_class]]
}

plt.figure()
plot_bar_chart(
    target_count.index.to_list(), target_count.to_list(), title="Class balance"
)
plt.show()

"""
Results:
Minority class= Yes : 110877
Majority class= No : 270055
Proportion: 0.41 : 1

As we can see, the dataset is unbalanced, because the difference between the frequency for Yes and No is higher than 0.5.
"""

# Separating the minority and majority classes
df_positives: Series = pos_covid_data[pos_covid_data[target] == positive_class]
df_negatives: Series = pos_covid_data[pos_covid_data[target] == negative_class]

# ------------------
# Approach 1: Undersampling: Randomly removing samples from the majority class (negative)
# ------------------

df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
df_under.to_csv(f"../../data/{pos_covid_file_tag}_under.csv", index=False)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")

"""
Results:
Minority class= Yes : 110877
Majority class= No : 110877
Proportion: 1.0 : 1
"""

# ------------------
# Approach 2: Oversampling
# ------------------

df_pos_sample: DataFrame = DataFrame(df_positives.sample(len(df_negatives), replace=True))
df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
df_over.to_csv(f"../../data/{pos_covid_file_tag}_over.csv", index=False)

print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(df_negatives))
print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")

"""
Results:
Minority class= Yes : 270055
Majority class= No : 270055
Proportion: 1.0 : 1
"""

# ------------------
# Approach 3: SMOTE
# ------------------


RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = pos_covid_data.pop(target).values
X: ndarray = pos_covid_data.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(pos_covid_data.columns) + [target]
df_smote.to_csv(f"data/{pos_covid_file_tag}_smote.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)
