from numpy import ndarray
from pandas import read_csv, DataFrame, concat
from sklearn.impute import SimpleImputer, KNNImputer

from utils.dslabs_functions import get_variable_types

pos_covid_filename: str = "../../data/class_pos_covid.csv"
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
print(f"Dataset nr records={pos_covid_data.shape[0]}", f"nr variables={pos_covid_data.shape[1]}")

# ------------------
# Approach 1: Dropping Missing Values
# ------------------

# Delete records with at least one missing value
df: DataFrame = pos_covid_data.dropna(how="any", inplace=False)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {pos_covid_data.shape[0] - df.shape[0]} records")

# 143302 records dropped - Bad approach because we lose too much data

# Delete only records where all values are missing
df: DataFrame = pos_covid_data.dropna(how="all", inplace=False)
print(f"Dataset nr records={df.shape[0]}",
      f"nr variables={df.shape[1]} after dropping {pos_covid_data.shape[0] - df.shape[0]} records")

# 0 records dropped - Bad approach because does not solve the problem

# Another approach is to delete variables with too many missing values, like TeatanusLast10Tdap or PneumoVaxEver, but
# this is not a good approach because we lose too much information

# TODO: Another option might be to only drop variables or records that show valid values above a given threshold - through the use of the thresh parameter.

"""
def mvi_by_dropping(
    data: DataFrame, min_pct_per_var: float = 0.1, min_pct_per_rec: float = 0.0
) -> DataFrame:
    # Deleting variables
    df: DataFrame = data.dropna(
        axis=1, thresh=data.shape[0] * min_pct_per_var, inplace=False
    )
    # Deleting records
    df.dropna(axis=0, thresh=data.shape[1] * min_pct_per_rec, inplace=True)

    return df


df: DataFrame = mvi_by_dropping(data, min_pct_per_variable=0.7, min_pct_per_record=0.9)
print(df.shape)
"""


# ------------------
# Approach 2: Imputing Missing Values
# ------------------

# TODO: variable encoding had to be applied beforehand
# "frequent" strategy is preferred, because does not change the distribution of the variable, using mean and mode

def mvi_by_filling(data: DataFrame, strategy: str = "frequent") -> DataFrame:
    df: DataFrame
    variables: dict = get_variable_types(data)
    stg_num, v_num = "mean", -1
    stg_sym, v_sym = "most_frequent", "NA"
    stg_bool, v_bool = "most_frequent", False
    if strategy != "knn":
        lst_dfs: list = []
        if strategy == "constant":
            stg_num, stg_sym, stg_bool = "constant", "constant", "constant"
        if len(variables["numeric"]) > 0:
            imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
            tmp_nr = DataFrame(
                imp.fit_transform(data[variables["numeric"]]),
                columns=variables["numeric"],
            )
            lst_dfs.append(tmp_nr)
        if len(variables["symbolic"]) > 0:
            imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
            tmp_sb = DataFrame(
                imp.fit_transform(data[variables["symbolic"]]),
                columns=variables["symbolic"],
            )
            lst_dfs.append(tmp_sb)
        if len(variables["binary"]) > 0:
            imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
            tmp_bool = DataFrame(
                imp.fit_transform(data[variables["binary"]]),
                columns=variables["binary"],
            )
            lst_dfs.append(tmp_bool)
        df = concat(lst_dfs, axis=1)
    else:
        imp = KNNImputer(n_neighbors=5)
        imp.fit(data)
        ar: ndarray = imp.transform(data)
        df = DataFrame(ar, columns=data.columns, index=data.index)
    return df
