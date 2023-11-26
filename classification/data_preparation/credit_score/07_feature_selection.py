import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

from utils.dslabs_functions import HEIGHT, study_variance_for_feature_selection
from utils.dslabs_functions import (
    select_low_variance_variables,
)

credit_score_train_file: str = "../../data/credit_score/processed_data/class_credit_score_scaled_minmax.csv"  # After scaling
credit_score_test_file: str = "../../data/credit_score/processed_data/class_credit_score_test.csv"
credit_score_file_tag: str = "class_credit_score"
credit_score_train: DataFrame = read_csv(credit_score_train_file)
credit_score_test: DataFrame = read_csv(credit_score_test_file)
target: str = "CovidPos"

# ------------------
# Dropping Low Variance Variables
# ------------------

print("Original variables", len(credit_score_train.columns.to_list()), ":", credit_score_train.columns.to_list())
vars2drop: list[str] = select_low_variance_variables(credit_score_train, 3, target=target)
print("Variables to drop", len(vars2drop), ":", vars2drop)

eval_metric = "recall"

plt.figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    credit_score_train,
    credit_score_test,
    target=target,
    max_threshold=3,
    lag=0.1,
    metric=eval_metric,
    file_tag=credit_score_file_tag,
)
plt.show()
plt.clf()

# TODO: Not working: páscoa, arranja
