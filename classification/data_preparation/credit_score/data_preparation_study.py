import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

from utils.dslabs_functions import HEIGHT, study_variance_for_feature_selection, \
    study_redundancy_for_feature_selection
from utils.dslabs_functions import evaluate_approaches

credit_score_file_tag: str = "class_credit_score"
eval_metric = "accuracy"

run_sampling = False
sampling_amount = 0.001 if run_sampling else 1

run_mv_imputation_study = False
run_outliers_treatment_study = False
run_scaling_study = False
run_balancing_study = False
run_feature_selection_preliminary_study = True
run_feature_selection_study = True

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
            ["../../data/credit_score/processed_data/class_credit_score_imputed_mv_approach1.csv",
             "Approach 1 - Drop Records"],
            ["../../data/credit_score/processed_data/class_credit_score_imputed_mv_approach2.csv",
             "Approach 2 - Drop and Impute"]
        ],
        study_title="MV Imputation Study",
        metric=eval_metric,
        target="Credit_Score",
        file_tag=f"{credit_score_file_tag}_imputed_mv",
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
            ["../../data/credit_score/processed_data/class_credit_score_drop_outliers.csv",
             "Approach 1 - Drop Outliers"],
            ["../../data/credit_score/processed_data/class_credit_score_replacing_outliers.csv",
             "Approach 2 - Replace Outliers"],
            ["../../data/credit_score/processed_data/class_credit_score_truncate_outliers.csv",
             "Approach 3 - Truncate Outliers"]
        ],
        study_title="Outliers Treatment",
        metric=eval_metric,
        target="Credit_Score",
        file_tag=f"{credit_score_file_tag}_outliers",
        sample_amount=sampling_amount
    )


"""
------------------
Scaling (only KNN)
------------------
% Approach 1: Standard Scaler
% Approach 2: MinMax Scaler
"""
if run_scaling_study:
    evaluate_approaches(
        approaches=[
            ["../../data/credit_score/processed_data/class_credit_score_scaled_zscore.csv",
             "Approach 1 - Standard Scaler"],
            ["../../data/credit_score/processed_data/class_credit_score_scaled_minmax.csv",
             "Approach 2 - MinMax Scaler"]
        ],
        study_title="Scaling",
        metric=eval_metric,
        target="Credit_Score",
        file_tag=f"{credit_score_file_tag}_scaling",
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
            ["../../data/credit_score/processed_data/class_credit_score_train_under.csv", "Approach 1 - Undersampling"],
            ["../../data/credit_score/processed_data/class_credit_score_train_over.csv", "Approach 2 - Oversampling"],
            ["../../data/credit_score/processed_data/class_credit_score_train_smote.csv", "Approach 3 - SMOTE"]
        ],
        study_title="Balancing",
        metric=eval_metric,
        target="Credit_Score",
        file_tag=f"{credit_score_file_tag}_balancing",
        sample_amount=sampling_amount
    )

"""
------------------
Feature Selection Preliminary Study
------------------
"""
if run_feature_selection_preliminary_study:
    credit_score_train_file: str = "../../data/credit_score/processed_data/class_credit_score_train_over.csv"
    credit_score_test_file: str = "../../data/credit_score/processed_data/class_credit_score_test.csv"
    credit_score_train: DataFrame = read_csv(credit_score_train_file)
    credit_score_test: DataFrame = read_csv(credit_score_test_file)
    target: str = "Credit_Score"

    if run_sampling:
        credit_score_train = credit_score_train.sample(frac=sampling_amount, random_state=42)
        credit_score_test = credit_score_test.sample(frac=sampling_amount, random_state=42)

    eval_metric = "recall"

    plt.figure(figsize=(2 * HEIGHT, HEIGHT))
    study_variance_for_feature_selection(
        credit_score_train,
        credit_score_test,
        target=target,
        max_threshold=1.5,
        lag=0.1,
        metric=eval_metric,
        file_tag=credit_score_file_tag,
    )
    plt.show()
    plt.clf()

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
            ["../../data/credit_score/processed_data/class_credit_score_train_lowvar.csv", "Approach 1 - Dropping Low Variance Variables"],
            ["../../data/credit_score/processed_data/class_credit_score_train_redundant.csv", "Approach 2 - Dropping Highly Correlated Variables"],
        ],
        study_title="Feature Selection",
        metric=eval_metric,
        target="Credit_Score",
        file_tag=f"{credit_score_file_tag}_feature_selection",
        sample_amount=sampling_amount
    )