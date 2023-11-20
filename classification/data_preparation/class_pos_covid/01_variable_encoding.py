from numpy import ndarray
from pandas import DataFrame, read_csv, concat
from sklearn.preprocessing import OneHotEncoder

from utils.dslabs_functions import get_variable_types

"""
Variables encoding is the first step to apply, and it is only required in the presence of
symbolic variables. This operation shall result directly from the granularity analysis performed
in the data profiling step. Among the techniques available you find transforming into numeric
and dummification. Different choices have usually to be made for each variable, however
only a choice per variable shall be applied, without applying more than one alternative
"""

pos_covid_filename: str = "../../data/class_pos_covid.csv"
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")

# print binary variables and their values
vars: dict[str, list] = get_variable_types(pos_covid_data)
print(vars["binary"])
for v in vars["binary"]:
    print(v, pos_covid_data[v].unique())

print("--------------")
# print ordinal variables and their values
for v in vars["symbolic"]:
    print(v, pos_covid_data[v].unique())

# ------------------
# Ordinal Encoding
# ------------------

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
sex_type_values: dict[str, int] = {"Female": 0, "Male": 1}

general_health_type_values: dict[str, int] = {
    "Excellent": 0,
    "Very good": 1,
    "Good": 2,
    "Fair": 3,
    "Poor": 4,
}

# TODO: this does not make sense, but I think dummification is not a good idea, because it will create a lot of columns
states = ['Alabama' 'Alaska' 'Arizona' 'Arkansas' 'California' 'Colorado'
          'Connecticut' 'Delaware' 'District of Columbia' 'Florida' 'Georgia'
          'Hawaii' 'Idaho' 'Illinois' 'Indiana' 'Iowa' 'Kansas' 'Kentucky'
          'Louisiana' 'Maine' 'Maryland' 'Massachusetts' 'Michigan' 'Minnesota'
          'Mississippi' 'Missouri' 'Montana' 'Nebraska' 'Nevada' 'New Hampshire'
          'New Jersey' 'New Mexico' 'New York' 'North Carolina' 'North Dakota'
          'Ohio' 'Oklahoma' 'Oregon' 'Pennsylvania' 'Rhode Island' 'South Carolina'
          'South Dakota' 'Tennessee' 'Texas' 'Utah' 'Vermont' 'Virginia'
          'Washington' 'West Virginia' 'Wisconsin' 'Wyoming' 'Guam' 'Puerto Rico'
          'Virgin Islands']
state_type_values: dict[str, int] = {state: i for i, state in enumerate(states)}

encoding: dict[str, dict[str, int]] = {
    # Binary variables
    "Sex": sex_type_values,
    "PhysicalActivities": yes_no,
    "HadHeartAttack": yes_no,
    "HadAngina": yes_no,
    "HadStroke": yes_no,
    "HadAsthma": yes_no,
    "HadSkinCancer": yes_no,
    "HadCOPD": yes_no,
    "HadDepressiveDisorder": yes_no,
    "HadKidneyDisease": yes_no,
    "HadArthritis": yes_no,
    "DeafOrHardOfHearing": yes_no,
    "BlindOrVisionDifficulty": yes_no,
    "DifficultyConcentrating": yes_no,
    "DifficultyWalking": yes_no,
    "DifficultyDressingBathing": yes_no,
    "DifficultyErrands": yes_no,
    "ChestScan": yes_no,
    "AlcoholDrinkers": yes_no,
    "HIVTesting": yes_no,
    "FluVaxLast12": yes_no,
    "PneumoVaxEver": yes_no,
    "HighRiskLastYear": yes_no,
    "CovidPos": yes_no,
    # TODO: Finish encoding, and use Dummification
}
df: DataFrame = pos_covid_data.replace(encoding, inplace=False)
df.head()


# ------------------
# Dummification
# ------------------

def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    other_vars: list[str] = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype="bool", drop="if_binary"
    )
    trans: ndarray = enc.fit_transform(df[vars_to_dummify])

    new_vars: ndarray = enc.get_feature_names_out(vars_to_dummify)
    dummy = DataFrame(trans, columns=new_vars, index=df.index)

    final_df: DataFrame = concat([df[other_vars], dummy], axis=1)
    return final_df
