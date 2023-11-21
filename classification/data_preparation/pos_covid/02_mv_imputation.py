from pandas import read_csv, DataFrame

from utils.dslabs_functions import mvi_by_dropping, mvi_by_filling

pos_covid_filename: str = "../../data/pos_covid/processed_data/class_pos_covid_encoded.csv" # After encoding
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
print(f"Dataset nr records={pos_covid_data.shape[0]}", f"nr variables={pos_covid_data.shape[1]}")

# ------------------
# Approach 1: Dropping Missing Values
# ------------------

# Delete records with at least one missing value
print("Dropping records with at least one missing value")
df: DataFrame = pos_covid_data.dropna(how="any", inplace=False)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {pos_covid_data.shape[0] - df.shape[0]} records")

# 143302 records dropped - Bad approach because we lose too much data

# ------------------

# Delete only records where all values are missing
print("Dropping records where all values are missing")
df: DataFrame = pos_covid_data.dropna(how="all", inplace=False)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {pos_covid_data.shape[0] - df.shape[0]} records")

# 0 records dropped - Bad approach because does not solve the problem

# ------------------

# Another approach is to delete variables with too many missing values, like TeatanusLast10Tdap or PneumoVaxEver, but
# this is not a good approach because we lose too much information

# ------------------

# Drop variables or records that show valid values above a given threshold - through the use of the thresh parameter

print("Dropping variables with too many missing values")
df: DataFrame = mvi_by_dropping(pos_covid_data, min_pct_per_variable=0.7, min_pct_per_record=0.9)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {pos_covid_data.shape[0] - df.shape[0]} records")

# ------------------
# Approach 2: Imputing Missing Values
# ------------------

# "frequent" strategy is preferred, because does not change the distribution of the variable, using mean and mode

df: DataFrame = mvi_by_filling(pos_covid_data, strategy="frequent")
print(df.head(10))

# TODO: Analyze the results of the imputation

df.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_imputed_mv.csv", index=False)
