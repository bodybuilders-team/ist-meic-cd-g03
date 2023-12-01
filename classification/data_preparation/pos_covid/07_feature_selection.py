import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

from utils.dslabs_functions import HEIGHT, study_variance_for_feature_selection, study_redundancy_for_feature_selection
from utils.dslabs_functions import (
    select_low_variance_variables,
)

pos_covid_train_file: str = "../../data/pos_covid/processed_data/class_pos_covid_scaled_minmax.csv"  # After scaling
pos_covid_test_file: str = "../../data/pos_covid/processed_data/class_pos_covid_test.csv"
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_train: DataFrame = read_csv(pos_covid_train_file)
pos_covid_test: DataFrame = read_csv(pos_covid_test_file)
target: str = "CovidPos"

run_sampling = False
sampling_amount = 0.1

if run_sampling:
    pos_covid_train = pos_covid_train.sample(frac=sampling_amount, random_state=42)
    pos_covid_test = pos_covid_test.sample(frac=sampling_amount, random_state=42)

# ------------------
# Approach 1: Dropping Low Variance Variables
# ------------------

# print("Original variables", len(pos_covid_train.columns.to_list()), ":", pos_covid_train.columns.to_list())
# vars2drop: list[str] = select_low_variance_variables(pos_covid_train, 0.1, target=target)
# print("Variables to drop", len(vars2drop), ":", vars2drop)
#
# ------------------

eval_metric = "recall"

plt.figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    pos_covid_train,
    pos_covid_test,
    target=target,
    max_threshold=0.05,
    lag=0.002,
    metric=eval_metric,
    file_tag=pos_covid_file_tag,
)
plt.show()
plt.clf()


eval_metric = "recall"

plt.figure(figsize=(2 * HEIGHT, HEIGHT))
study_redundancy_for_feature_selection(
    pos_covid_train,
    pos_covid_test,
    target=target,
    min_threshold=0.10,
    lag=0.05,
    metric=eval_metric,
    file_tag=pos_covid_file_tag,
)
plt.show()