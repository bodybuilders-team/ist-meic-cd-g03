from utils.dslabs_functions import data_prep_naive_bayes_study

pos_covid_file_tag: str = "class_pos_covid"
eval_metric = "accuracy"

"""
------------------
MV Imputation
------------------

% Approach 1: Only delete records with at least one missing value
% Approach 2: Dropping by threshold and imputing missing values with knn strategy
"""
# data_prep_naive_bayes_study(
#     approaches=[
#         ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach1.csv", "Approach 1 - Drop Records"],
#         ["../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach2.csv", "Approach 2 - Drop and Impute"]
#     ],
#     study_title="MV Imputation",
#     metric=eval_metric,
#     target="CovidPos",
#     file_tag=f"{pos_covid_file_tag}_imputed_mv"
# )
# TODO: KNN study

"""
------------------
Outliers Treatment
------------------

% Approach 1: Dropping Outliers
% Approach 2: Replacing outliers with fixed value
% Approach 3: Truncating outliers
"""
data_prep_naive_bayes_study(
    approaches=[
        ["../../data/pos_covid/processed_data/class_pos_covid_drop_outliers.csv", "Approach 1 - Drop Outliers"],
        ["../../data/pos_covid/processed_data/class_pos_covid_replacing_outliers.csv", "Approach 2 - Replace Outliers"],
        ["../../data/pos_covid/processed_data/class_pos_covid_truncate_outliers.csv", "Approach 3 - Truncate Outliers"]
    ],
    metric=eval_metric,
    study_title="Outliers Treatment",
    target="CovidPos",
    file_tag=f"{pos_covid_file_tag}_outliers"
)
# TODO: KNN study


"""
------------------
Scaling (only KNN)
------------------
% Approach 1: Standard Scaler
% Approach 2: MinMax Scaler
"""
# TODO: KNN study


"""
------------------
Balancing
------------------
% Approach 1: Undersampling
% Approach 2: Oversampling
% Approach 3: SMOTE
"""
# TODO: NB and KNN study
