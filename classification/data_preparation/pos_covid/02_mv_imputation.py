from pandas import read_csv, DataFrame

from utils.dslabs_functions import mvi_by_dropping, mvi_by_filling

pos_covid_filename: str = "../../data/pos_covid/processed_data/class_pos_covid_encoded.csv"  # After encoding
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
print(f"Dataset nr records={pos_covid_data.shape[0]}", f"nr variables={pos_covid_data.shape[1]}\n")


variable_types: dict[str, list[str]] = {
    "binary": ["Sex", "PhysicalActivities", "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
               "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "DeafOrHardOfHearing",
               "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
               "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
               "HighRiskLastYear", "CovidPos"],
    "categorical": ["GeneralHealth", "LastCheckupTime", "RemovedTeeth", "HadDiabetes", "SmokerStatus",
                    "ECigaretteUsage", "AgeCategory", "TetanusLast10Tdap"]
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


# 143302 records dropped - Bad approach because we lose too much data

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

# Another approach is to delete variables with too many missing values, like TeatanusLast10Tdap or PneumoVaxEver, but
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


# TODO: Analyze the results of the imputation

# ------------------
# Approach 1: Only delete records with at least one missing value
# ------------------

print("Approach 1: Only delete records with at least one missing value")
data_frame: DataFrame = delete_records_with_any_mv(pos_covid_data)

print("Saving to file...")
data_frame.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_imputed_mv_approach1.csv", index=False)
print("Saved to file.")

print()

# ------------------
# Approach 2: Dropping by threshold and imputing missing values with frequent strategy
# ------------------

print("Approach 2: Dropping by threshold and imputing missing values with frequent strategy")
df_drop_threshold: DataFrame = drop_data_by_mv_threshold(pos_covid_data, 0.7, 0.9)
data_frame: DataFrame = impute_mv(df_drop_threshold, "frequent")

print("Saving to file...")
data_frame.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_imputed_mv_approach2.csv", index=False)
print("Saved to file.")
