import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

from utils.dslabs_functions import HEIGHT, study_variance_for_feature_selection, study_redundancy_for_feature_selection
from utils.dslabs_functions import evaluate_approaches

pos_covid_file_tag: str = "class_pos_covid"
eval_metric = "accuracy"

run_sampling = False
sampling_amount = 0.01 if run_sampling else 1

run_mv_imputation_study = True
run_outliers_treatment_study = True
run_scaling_study = True
run_balancing_study = True
run_feature_selection_preliminary_study = True
run_feature_selection_study = True

sample_tag = f"_1_{int(1 / sampling_amount)}th" if run_sampling else ""

"""
------------------
MV Imputation
------------------

% Approach 1: Only delete records with at least one missing value
% Approach 2: Dropping by threshold and imputing missing values with knn strategy
"""
if run_mv_imputation_study:
    evaluate_approaches(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach1.csv",
             "Approach 1 - Drop Records"],
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach2.csv",
             "Approach 2 - Drop and Impute"]
        ],
        study_title="MV Imputation Study",
        metric=eval_metric,
        target="CovidPos",
        save_fig_path=f"images/{pos_covid_file_tag}_imputed_mv_eval{sample_tag}.png",
        sample_amount=sampling_amount
    )

"""
------------------
Outliers Treatment
------------------

% Approach 1: Dropping Outliers
% Approach 2: Replacing outliers with fixed value
% Approach 3: Truncating outliers
"""
if run_outliers_treatment_study:
    evaluate_approaches(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach2.csv",
             "Control"],
            ["../../data/pos_covid/processed_data/class_pos_covid_drop_outliers.csv",
             "Approach 1 - Drop Outliers"],
            ["../../data/pos_covid/processed_data/class_pos_covid_replacing_outliers.csv",
             "Approach 2 - Replace Outliers"],
            ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv",
             "Approach 3 - Truncate Outliers"]
        ],
        study_title="Outliers Treatment",
        metric=eval_metric,
        target="CovidPos",
        save_fig_path=f"images/{pos_covid_file_tag}_outliers_eval{sample_tag}.png",
        sample_amount=sampling_amount
    )

"""
------------------
Scaling (Only KNN)
------------------
% Approach 1: Standard Scaler
% Approach 2: MinMax Scaler
"""
if run_scaling_study:
    evaluate_approaches(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv",
             "Control"],
            ["../../data/pos_covid/processed_data/class_pos_covid_scaled_zscore.csv",
             "Approach 1 - Standard Scaler"],
            ["../../data/pos_covid/processed_data/class_pos_covid_scaled_minmax.csv",
             "Approach 2 - MinMax Scaler"]
        ],
        study_title="Scaling",
        metric=eval_metric,
        target="CovidPos",
        save_fig_path=f"images/{pos_covid_file_tag}_scaling_eval{sample_tag}.png",
        sample_amount=sampling_amount,
        nb=False
    )

"""
------------------
Balancing
------------------
% Approach 1: Undersampling
% Approach 2: Oversampling
% Approach 3: SMOTE
"""
if run_balancing_study:
    evaluate_approaches(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_scaled_zscore.csv",
             "Control"],
            ["../../data/pos_covid/processed_data/class_pos_covid_train_under.csv", "Approach 1 - Undersampling"],
            ["../../data/pos_covid/processed_data/class_pos_covid_train_over.csv", "Approach 2 - Oversampling"],
            ["../../data/pos_covid/processed_data/class_pos_covid_train_smote.csv", "Approach 3 - SMOTE"]
        ],
        study_title="Balancing",
        metric=eval_metric,
        target="CovidPos",
        save_fig_path=f"images/{pos_covid_file_tag}_balancing_eval{sample_tag}.png",
        sample_amount=sampling_amount
    )

"""
------------------
Feature Selection Preliminary Study
------------------
"""
if run_feature_selection_preliminary_study:
    pos_covid_train_file: str = "../../data/pos_covid/processed_data/class_pos_covid_train_over.csv"  # After balancing
    pos_covid_test_file: str = "../../data/pos_covid/processed_data/class_pos_covid_test.csv"
    pos_covid_train: DataFrame = read_csv(pos_covid_train_file)
    pos_covid_test: DataFrame = read_csv(pos_covid_test_file)
    target: str = "CovidPos"

    if run_sampling:
        pos_covid_train = pos_covid_train.sample(frac=sampling_amount, random_state=42)
        pos_covid_test = pos_covid_test.sample(frac=sampling_amount, random_state=42)

    eval_metric = "recall"

    plt.figure(figsize=(2 * HEIGHT, HEIGHT))
    study_variance_for_feature_selection(
        pos_covid_train,
        pos_covid_test,
        target=target,
        max_threshold=1.5,
        lag=0.1,
        metric=eval_metric,
        file_tag=pos_covid_file_tag,
        save_fig_path=f"images/{pos_covid_file_tag}_fs_low_var_{eval_metric}_study{sample_tag}.png"
    )
    plt.show()
    plt.clf()

    plt.figure(figsize=(2 * HEIGHT, HEIGHT))
    study_redundancy_for_feature_selection(
        pos_covid_train,
        pos_covid_test,
        target=target,
        min_threshold=0.10,
        lag=0.05,
        metric=eval_metric,
        file_tag=pos_covid_file_tag,
        save_fig_path=f"images/{pos_covid_file_tag}_fs_redundancy_{eval_metric}_study{sample_tag}.png"
    )
    plt.show()

"""
------------------
Feature Selection Study
------------------
% Approach 1: Dropping Low Variance Variables
% Approach 2: Dropping Highly Correlated Variables
"""
if run_feature_selection_study:
    evaluate_approaches(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_train_lowvar.csv",
             "Approach 1 - Dropping Low Variance Variables"],
            ["../../data/pos_covid/processed_data/class_pos_covid_train_redundant.csv",
             "Approach 2 - Dropping Highly Correlated Variables"],
        ],
        study_title="Feature Selection",
        metric=eval_metric,
        target="CovidPos",
        save_fig_path=f"images/{pos_covid_file_tag}_feature_selection_eval{sample_tag}.png",
        sample_amount=sampling_amount
    )
