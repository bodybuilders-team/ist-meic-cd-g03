import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

from utils.dslabs_functions import HEIGHT, study_variance_for_feature_selection, study_redundancy_for_feature_selection, \
    apply_feature_selection, select_redundant_variables
from utils.dslabs_functions import (
    select_low_variance_variables,
)

credit_score_train_file: str = "../../data/credit_score/processed_data/class_credit_score_scaled_minmax.csv"  # After scaling
credit_score_test_file: str = "../../data/credit_score/processed_data/class_credit_score_test.csv"
credit_score_file_tag: str = "class_credit_score"
credit_score_train: DataFrame = read_csv(credit_score_train_file)
credit_score_test: DataFrame = read_csv(credit_score_test_file)
target: str = "Credit_Score"

run_sampling = False
sampling_amount = 0.1

if run_sampling:
    credit_score_train = credit_score_train.sample(frac=sampling_amount, random_state=42)
    credit_score_test = credit_score_test.sample(frac=sampling_amount, random_state=42)

# ------------------
# Dropping Low Variance Variables
# ------------------

# print("Original variables", len(credit_score_train.columns.to_list()), ":", credit_score_train.columns.to_list())
# vars2drop: list[str] = select_low_variance_variables(credit_score_train, 3, target=target)
# print("Variables to drop", len(vars2drop), ":", vars2drop)

eval_metric = "recall"

plt.figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    credit_score_train,
    credit_score_test,
    target=target,
    max_threshold=0.1,
    lag=0.01,
    metric=eval_metric,
    file_tag=credit_score_file_tag,
)
plt.show()
plt.clf()

vars2drop: list[str] = select_low_variance_variables(
    credit_score_train, max_threshold=0.045, target=target
)
train_cp, test_cp = apply_feature_selection(
    credit_score_train, credit_score_test, vars2drop, filename=f"../../data/credit_score/processed_data/{credit_score_file_tag}_lowvar.csv", tag="lowvar"
)
print(f"Original data: train={credit_score_train.shape}, test={credit_score_test.shape}")
print(f"After low variance FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")

eval_metric = "recall"

plt.figure(figsize=(2 * HEIGHT, HEIGHT))
study_redundancy_for_feature_selection(
    credit_score_train,
    credit_score_test,
    target=target,
    min_threshold=0.10,
    lag=0.05,
    metric=eval_metric,
    file_tag=credit_score_file_tag,
)
plt.show()

vars2drop: list[str] = select_redundant_variables(
    credit_score_train, min_threshold=0.065, target=target
)
train_cp, test_cp = apply_feature_selection(
    credit_score_train, credit_score_test, vars2drop, filename=f"../../data/credit_score/processed_data/{credit_score_file_tag}_redundant.csv", tag="redundant"
)
print(f"Original data: train={credit_score_train.shape}, test={credit_score_test.shape}")
print(f"After redundant FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")