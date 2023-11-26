from pandas import read_csv, DataFrame

from utils.dslabs_functions import mvi_by_dropping, mvi_by_filling

credit_score_filename: str = "../../data/credit_score/processed_data/class_credit_score_encoded.csv"  # After encoding
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="") # TODO , index_col="ID" - ID column was removed after encoding and I don't know why
print(f"Dataset nr records={credit_score_data.shape[0]}", f"nr variables={credit_score_data.shape[1]}")

# ------------------
# Approach 1: Dropping Missing Values
# ------------------

# Delete records with at least one missing value
print("Dropping records with at least one missing value")
df: DataFrame = credit_score_data.dropna(how="any", inplace=False)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {credit_score_data.shape[0] - df.shape[0]} records")

# 63755 records dropped - Bad approach because we lose too much data

# ------------------

# Delete only records where all values are missing
print("Dropping records where all values are missing")
df: DataFrame = credit_score_data.dropna(how="all", inplace=False)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {credit_score_data.shape[0] - df.shape[0]} records")

# 0 records dropped - Bad approach because does not solve the problem

# ------------------

# Another approach is to delete variables with too many missing values, but
# this is not a good approach because we lose too much information

# ------------------

# Drop variables or records that show valid values above a given threshold - through the use of the thresh parameter

print("Dropping variables with too many missing values")
df: DataFrame = mvi_by_dropping(credit_score_data, min_pct_per_variable=0.7, min_pct_per_record=0.9)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {credit_score_data.shape[0] - df.shape[0]} records")

# ------------------
# Approach 2: Imputing Missing Values
# ------------------

# "frequent" strategy is preferred, because does not change the distribution of the variable, using mean and mode

df: DataFrame = mvi_by_filling(credit_score_data, strategy="frequent")
print(df.head(10))

# TODO: Analyze the results of the imputation

df.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_imputed_mv.csv", index=False)
