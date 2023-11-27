from utils.dslabs_functions import data_prep_knn_study, data_prep_naive_bayes_study

pos_covid_file_tag: str = "class_pos_covid"
eval_metric = "accuracy"

run_sampling = True
sampling_amount = 0.01 if run_sampling else 1

run_mv_imputation_study = False
run_outliers_treatment_study = False
run_scaling_study = False
run_balancing_study = True

"""
------------------
MV Imputation
------------------

% Approach 1: Only delete records with at least one missing value
% Approach 2: Dropping by threshold and imputing missing values with knn strategy
"""
if run_mv_imputation_study:
    data_prep_naive_bayes_study(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach1.csv",
             "Approach 1 - Drop Records"],
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach2.csv",
             "Approach 2 - Drop and Impute"]
        ],
        study_title="MV Imputation",
        metric=eval_metric,
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_imputed_mv",
        sampling_amount=sampling_amount
    )
    data_prep_knn_study(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach1.csv",
             "Approach 1 - Drop Records"],
            ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach2.csv",
             "Approach 2 - Drop and Impute"]
        ],
        study_title="MV Imputation",
        metric=eval_metric,
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_imputed_mv",
        sampling_amount=sampling_amount
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
    data_prep_naive_bayes_study(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_drop_outliers.csv", "Approach 1 - Drop Outliers"],
            ["../../data/pos_covid/processed_data/class_pos_covid_replacing_outliers.csv",
             "Approach 2 - Replace Outliers"],
            ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv",
             "Approach 3 - Truncate Outliers"]
        ],
        metric=eval_metric,
        study_title="Outliers Treatment",
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_outliers",
        sampling_amount=sampling_amount
    )
    data_prep_knn_study(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_drop_outliers.csv", "Approach 1 - Drop Outliers"],
            ["../../data/pos_covid/processed_data/class_pos_covid_replacing_outliers.csv",
             "Approach 2 - Replace Outliers"],
            ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv",
             "Approach 3 - Truncate Outliers"]
        ],
        metric=eval_metric,
        study_title="Outliers Treatment",
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_outliers",
        sampling_amount=sampling_amount
    )


"""
------------------
Scaling (only KNN)
------------------
% Approach 1: Standard Scaler
% Approach 2: MinMax Scaler
"""
if run_scaling_study:
    data_prep_knn_study( # TODO: Fix this
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_scaled_zscore.csv", "Approach 1 - Standard Scaler"],
            ["../../data/pos_covid/processed_data/class_pos_covid_scaled_minmax.csv", "Approach 2 - MinMax Scaler"]
        ],
        metric=eval_metric,
        study_title="Scaling",
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_scaling",
        sampling_amount=sampling_amount
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
    data_prep_naive_bayes_study(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_drop_outliers.csv", "Approach 1 - Undersampling"],
            ["../../data/pos_covid/processed_data/class_pos_covid_replacing_outliers.csv", "Approach 2 - Oversampling"],
            ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv", "Approach 3 - SMOTE"]
        ],
        metric=eval_metric,
        study_title="Balancing",
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_balancing",
        sampling_amount=sampling_amount
    )
    data_prep_knn_study(
        approaches=[
            ["../../data/pos_covid/processed_data/class_pos_covid_drop_outliers.csv", "Approach 1 - Undersampling"],
            ["../../data/pos_covid/processed_data/class_pos_covid_replacing_outliers.csv", "Approach 2 - Oversampling"],
            ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv", "Approach 3 - SMOTE"]
        ],
        metric=eval_metric,
        study_title="Balancing",
        target="CovidPos",
        file_tag=f"{pos_covid_file_tag}_balancing",
        sampling_amount=sampling_amount
    )
