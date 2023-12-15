from pandas import read_csv, DataFrame, Series

from utils.dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

credit_score_filename: str = "../../data/credit_score/processed_data/class_credit_score_imputed_mv_approach2.csv"  # After imputation
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename)
print(f"Dataset nr records={credit_score_data.shape[0]}", f"nr variables={credit_score_data.shape[1]}")

# ------------------
# Approach 1: Dropping Outliers
# ------------------

n_std: int = NR_STDEV
numeric_vars: list[str] = get_variable_types(credit_score_data)["numeric"]
if numeric_vars is not None:
    df: DataFrame = credit_score_data.copy(deep=True)
    summary5: DataFrame = credit_score_data[numeric_vars].describe()
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary5[var]
        )
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    df.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_drop_outliers.csv", index=False)
    print(f"Data after dropping outliers: {df.shape[0]} records and {df.shape[1]} variables")
else:
    print("There are no numeric variables")

# Dropped 31105 records

# ------------------
# Approach 2: Replacing outliers with fixed value
# ------------------

# Instead of dropping all the records with outliers, it is also possible to replace the outliers with a fixed value,
# for example its median value.

if numeric_vars:
    df: DataFrame = credit_score_data.copy(deep=True)
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
    df.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_replacing_outliers.csv", index=False)
    print(f"Data after replacing outliers: {df.shape[0]} records and {df.shape[1]} variables")
    print(df.describe())
else:
    print("There are no numeric variables")

# ------------------
# Approach 3: Truncating outliers
# ------------------

# Another possibility is to truncate the outliers to the minimum/maximum accepted as regular objects

if numeric_vars:
    df: DataFrame = credit_score_data.copy(deep=True)
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        df[var] = df[var].apply(
            lambda x: top if x > top else bottom if x < bottom else x
        )
    df.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_truncate_outliers.csv", index=False)
    print("Data after truncating outliers:", df.shape)
    print(df.describe())
else:
    print("There are no numeric variables")
