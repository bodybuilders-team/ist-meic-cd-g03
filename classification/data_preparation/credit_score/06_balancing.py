import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from pandas import read_csv, DataFrame, Series, concat

from utils.dslabs_functions import plot_bar_chart

"""
A dataset is unbalanced if the number of samples in each class is not similar - this can bias the model.
"""

credit_score_filename: str = "../../data/credit_score/processed_data/class_credit_score_train.csv"  # Only to the train set
credit_score_file_tag: str = "class_credit_score_train"
credit_score_data: DataFrame = read_csv(credit_score_filename)
target: str = "Credit_Score"

target_count: Series = credit_score_data[target].value_counts()
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
plt.savefig(f"images/{credit_score_file_tag}_class_balance.png")
plt.show()
plt.clf()

"""
Results:
?
"""

# Separating the minority and majority classes
df_positives: Series = credit_score_data[credit_score_data[target] == positive_class]
df_negatives: Series = credit_score_data[credit_score_data[target] == negative_class]

# ------------------
# Approach 1: Undersampling: Randomly removing samples from the majority class (negative)
# ------------------

df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
df_under.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_under.csv", index=False)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")

"""
Results:
?
"""

# ------------------
# Approach 2: Oversampling
# ------------------

df_pos_sample: DataFrame = DataFrame(df_positives.sample(len(df_negatives), replace=True))
df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
df_over.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_over.csv", index=False)

print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(df_negatives))
print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")

"""
Results:
?
"""

# ------------------
# Approach 3: SMOTE
# ------------------


RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = credit_score_data.pop(target).values
X: ndarray = credit_score_data.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(credit_score_data.columns) + [target]
df_smote.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_smote.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)
