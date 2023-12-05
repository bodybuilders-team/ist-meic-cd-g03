from pandas import DataFrame, read_csv

from utils.dslabs_functions import (
    select_low_variance_variables,
)
from utils.dslabs_functions import select_redundant_variables, apply_feature_selection

pos_covid_train_file: str = "../../data/pos_covid/processed_data/class_pos_covid_train_over.csv"  # After balancing
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

vars2drop: list[str] = select_low_variance_variables(
    pos_covid_train, max_threshold=0.8, target=target
)
train_cp, test_cp = apply_feature_selection(
    pos_covid_train, pos_covid_test, vars2drop, filename=f"../../data/pos_covid/processed_data/{pos_covid_file_tag}", tag="lowvar"
)
print(f"Original data: train={pos_covid_train.shape}, test={pos_covid_test.shape}")
print(f"After low variance FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")

# ------------------
# Approach 2: Dropping Highly Correlated Variables
# ------------------

vars2drop: list[str] = select_redundant_variables(
    pos_covid_train, min_threshold=0.45, target=target
)
train_cp, test_cp = apply_feature_selection(
    pos_covid_train, pos_covid_test, vars2drop, filename=f"../../data/pos_covid/processed_data/{pos_covid_file_tag}", tag="redundant"
)
print(f"Original data: train={pos_covid_train.shape}, test={pos_covid_train.shape}")
print(f"After redundant FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")