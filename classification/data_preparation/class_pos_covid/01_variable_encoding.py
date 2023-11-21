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

pos_covid_filename: str = "../../data/pos_covid/class_pos_covid.csv"
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

# I think dummification is not a good idea, because it will create a lot of columns
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

last_checkup_time_type_values: dict[str, int] = {
    "Within past year (anytime less than 12 months ago)": 0,
    "Within past 2 years (1 year but less than 2 years ago)": 1,
    "Within past 5 years (2 years but less than 5 years ago)": 2,
    "5 or more years ago": 3,
}

removed_teeth_type_values: dict[str, int] = {
    "None of them": 0,
    "1 to 5": 1,
    "6 or more, but not all": 2,
    "All": 3,
}

had_diabetes_type_values: dict[str, int] = {
    "No": 0,
    "No, pre-diabetes or borderline diabetes": 1,
    "Yes": 2,
    "Yes, but only during pregnancy (female)": 3
}

smoker_status_type_values: dict[str, int] = {
    "Never smoked": 0,
    "Current smoker - now smokes some days": 1,
    "Former smoker": 2,
    "Current smoker - now smokes every day": 3
}

e_cigarette_usage_type_values: dict[str, int] = {
    "Never used e-cigarettes in my entire life": 0,
    "Not at all (right now)": 1,
    "Use them some days": 2,
    "Use them every day": 3
}

age_category_type_values: dict[str, int] = {
    "Age 18 to 24": 0,
    "Age 25 to 29": 1,
    "Age 30 to 34": 2,
    "Age 35 to 39": 3,
    "Age 40 to 44": 4,
    "Age 45 to 49": 5,
    "Age 50 to 54": 6,
    "Age 55 to 59": 7,
    "Age 60 to 64": 8,
    "Age 65 to 69": 9,
    "Age 70 to 74": 10,
    "Age 75 to 79": 11,
    "Age 80 or older": 12,
}

tetaus_last_10_tdap_type_values: dict[str, int] = {
    "No, did not receive any tetanus shot in the past 10 years": 0,
    "Yes, received tetanus shot, but not Tdap": 1,
    "Yes, received Tdap": 2,
    "Yes, received tetanus shot but not sure what type": 3
}

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
    # Categorical variables
    "GeneralHealth": general_health_type_values,
    "State": state_type_values,
    "LastCheckupTime": last_checkup_time_type_values,
    "RemovedTeeth": removed_teeth_type_values,
    "HadDiabetes": had_diabetes_type_values,
    "SmokerStatus": smoker_status_type_values,
    "ECigaretteUsage": e_cigarette_usage_type_values,
    "AgeCategory": age_category_type_values,
    "TetausLast10Tdap": tetaus_last_10_tdap_type_values
}
df: DataFrame = pos_covid_data.replace(encoding, inplace=False)
df.head()


# ------------------
# Dummification
# ------------------

# When after exploring all possible perspectives, we are not able to specify an acceptable order among those variables
# the only solution is dummification : RaceEthnicityCategory

# RaceEthnicityCategory ['White only, Non-Hispanic' 'Black only, Non-Hispanic'
#  'Multiracial, Non-Hispanic' nan 'Hispanic'
#  'Other race only, Non-Hispanic']

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


df = dummify(df, ["RaceEthnicityCategory"])
df.head()
