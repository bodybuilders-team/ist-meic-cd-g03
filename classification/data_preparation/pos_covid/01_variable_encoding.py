from pandas import DataFrame, read_csv

from utils.dslabs_functions import encode_cyclic_variables

pos_covid_filename: str = "../../data/pos_covid/class_pos_covid.csv"
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")

# ------------------
# Ordinal Encoding: Binary and Categorical variables with order
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
    "Yes, but only during pregnancy (female)": 2,
    "Yes": 3
}

smoker_status_type_values: dict[str, int] = {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker - now smokes some days": 2,
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

tetanus_last_10_tdap_type_values: dict[str, int] = {
    "No, did not receive any tetanus shot in the past 10 years": 0,
    "Yes, received tetanus shot, but not Tdap": 1,
    "Yes, received tetanus shot but not sure what type": 2,
    "Yes, received Tdap": 3
}

race_ethnicity_values: dict[str, float] = {
    'White only, Non-Hispanic': 0.0,  # full white
    'Hispanic': 0.25,  # close to white
    'Other race only, Non-Hispanic': 0.4,  # probably asian or native
    'Multiracial, Non-Hispanic': 0.5,  # half white, half black
    'Black only, Non-Hispanic': 1.0,  # full black
}

state_type_values_transform_1: dict[str, str] = {
    "Alabama": "South",
    "Alaska": "West",
    "Arizona": "West",
    "Arkansas": "South",
    "California": "West",
    "Colorado": "West",
    "Connecticut": "Northeast",
    "Delaware": "South",
    "Florida": "South",
    "Georgia": "South",
    "Hawaii": "West",
    "Idaho": "West",
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Iowa": "Midwest",
    "Kansas": "Midwest",
    "Kentucky": "South",
    "Louisiana": "South",
    "Maine": "Northeast",
    "Maryland": "South",
    "Massachusetts": "Northeast",
    "Michigan": "Midwest",
    "Minnesota": "Midwest",
    "Mississippi": "South",
    "Missouri": "Midwest",
    "Montana": "West",
    "Nebraska": "Midwest",
    "Nevada": "West",
    "New Hampshire": "Northeast",
    "New Jersey": "Northeast",
    "New Mexico": "West",
    "New York": "Northeast",
    "North Carolina": "South",
    "North Dakota": "Midwest",
    "Ohio": "Midwest",
    "Oklahoma": "South",
    "Oregon": "West",
    "Pennsylvania": "Northeast",
    "Rhode Island": "Northeast",
    "South Carolina": "South",
    "South Dakota": "Midwest",
    "Tennessee": "South",
    "Texas": "South",
    "Utah": "West",
    "Vermont": "Northeast",
    "Virginia": "South",
    "Washington": "West",
    "West Virginia": "South",
    "Wisconsin": "Midwest",
    "Wyoming": "West",
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
    # Categorical variables with order
    "GeneralHealth": general_health_type_values,
    "State": state_type_values_transform_1,
    "LastCheckupTime": last_checkup_time_type_values,
    "RemovedTeeth": removed_teeth_type_values,
    "HadDiabetes": had_diabetes_type_values,
    "SmokerStatus": smoker_status_type_values,
    "ECigaretteUsage": e_cigarette_usage_type_values,
    "AgeCategory": age_category_type_values,
    "TetanusLast10Tdap": tetanus_last_10_tdap_type_values,
    "RaceEthnicityCategory": race_ethnicity_values
}
df: DataFrame = pos_covid_data.replace(encoding, inplace=False)

# Now that the State values were transformed, they will be encoded with a cyclic technique.
# Also, the records that do not correspond to the 4 possible regions, will have the State value changes
# to an empty string

# List of valid state values
valid_states = ['Northeast', 'South', 'West', 'Midwest']

# Replace non-valid state values with an empty string
df['Region'] = df['State'].replace({state: '' for state in df['State'].unique() if state not in valid_states})
df = df.drop('State', axis=1)

state_type_values_transform_2: dict[str, float] = {
    "Northeast": 1,
    "West": 2,
    "South": 3,
    "Midwest": 4
}

encoding: dict[str, dict[str, float]] = {"Region": state_type_values_transform_2}

df: DataFrame = df.replace(encoding, inplace=False)
encode_cyclic_variables(df, ['Region'])
df = df.drop('Region', axis=1)

# states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
#           'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
#           'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
#           'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
#           'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
#           'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
#           'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
#           'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
#           'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Guam', 'Puerto Rico',
#           'Virgin Islands'
#           ]
# state_type_values: dict[str, int] = {state: i for i, state in enumerate(states)}

# State using label encoder
# le = LabelEncoder()
# df['State'] = le.fit_transform(df['State'])

# ------------------
# Dummification
# ------------------

# When after exploring all possible perspectives, we are not able to specify an acceptable order among those variables
# the only solution is dummification : RaceEthnicityCategory

# RaceEthnicityCategory ['White only, Non-Hispanic' 'Black only, Non-Hispanic'
#  'Multiracial, Non-Hispanic' nan 'Hispanic'
#  'Other race only, Non-Hispanic']

# df = dummify(df, ["RaceEthnicityCategory"])
# #print(df.head(5))
#
df.to_csv(f"../../data/pos_covid/processed_data/{pos_covid_file_tag}_encoded.csv", index=False)
