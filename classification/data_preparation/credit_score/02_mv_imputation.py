from pandas import read_csv, DataFrame

from utils.dslabs_functions import mvi_by_dropping, mvi_by_filling

credit_score_filename: str = "../../data/credit_score/processed_data/class_credit_score_encoded.csv"  # After encoding
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename,
                                        na_values="")  # TODO , index_col="ID" - ID column was removed after encoding and I don't know why
print(f"Dataset nr records={credit_score_data.shape[0]}", f"nr variables={credit_score_data.shape[1]}")

variable_types: dict[str, list[str]] = {
    "binary": ["Credit_Score"],
    "categorical": ["CreditMix", "Payment_of_Min_Amount", "Payment_Behaviour"]
}


# ------------------
# Delete records with at least one missing value
# ------------------


def delete_records_with_any_mv(df: DataFrame) -> DataFrame:
    print("Dropping records with at least one missing value")
    df1: DataFrame = df.dropna(how="any", inplace=False)
    print(f"Dataset nr records={df1.shape[0]}",
          f"nr variables={df1.shape[1]} after dropping {df.shape[0] - df1.shape[0]} records")

    return df1


# 63755 records dropped - Bad approach because we lose too much data

# ------------------
# Delete only records where all values are missing
# ------------------

def delete_records_with_all_mv(df: DataFrame) -> DataFrame:
    print("Dropping records where all values are missing")
    df1: DataFrame = df.dropna(how="all", inplace=False)
    print(f"Dataset nr records={df1.shape[0]}",
          f"nr variables={df1.shape[1]} after dropping {df.shape[0] - df1.shape[0]} records")

    return df1


# 0 records dropped - Bad approach because does not solve the problem

# ------------------

# Another approach is to delete variables with too many missing values, but
# this is not a good approach because we lose too much information

# ------------------
# Drop variables or records that show valid values above a given threshold - through the use of the thresh parameter
# ------------------

def drop_data_by_mv_threshold(df: DataFrame, threshold_per_variable: float,
                              threshold_per_record: float) -> DataFrame:
    print(f"Dropping variables by threshold: {threshold_per_variable} and records by threshold: {threshold_per_record}")
    df1: DataFrame = mvi_by_dropping(df, min_pct_per_variable=threshold_per_variable,
                                     min_pct_per_record=threshold_per_record)
    print(f"Dataset nr records={df1.shape[0]}",
          f"nr variables={df1.shape[1]} after dropping {df.shape[0] - df1.shape[0]} records")

    return df1


# ------------------
# Imputing Missing Values
# ------------------

# "frequent" strategy is preferred, because does not change the distribution of the variable, using mean and mode

def impute_mv(df: DataFrame, strategy: str) -> DataFrame:
    print(f"Imputing missing values by strategy: {strategy}")
    df1: DataFrame = mvi_by_filling(df, strategy=strategy, variable_types=variable_types)

    return df1


# ------------------
# Approach 1: Only delete records with at least one missing value
# ------------------

print("Approach 1: Only delete records with at least one missing value")
data_frame: DataFrame = delete_records_with_any_mv(credit_score_data)

print("Saving to file...")
data_frame.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_imputed_mv_approach1.csv", index=False)
print("Saved to file.")

print()

# ------------------
# Approach 2: Dropping by threshold and imputing missing values with frequent strategy
# ------------------

print("Approach 2: Dropping by threshold and imputing missing values with frequent strategy")
df_drop_threshold: DataFrame = drop_data_by_mv_threshold(credit_score_data, 0.7, 0.9)
data_frame: DataFrame = impute_mv(df_drop_threshold, "frequent")

print("Saving to file...")
data_frame.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_imputed_mv_approach2.csv", index=False)
print("Saved to file.")
