import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import analyse_property_granularity

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values=['', float('nan')])

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values='', index_col="ID")


# ------------------
# Granularity analysis for dataset "class_pos_covid"
# ------------------

# Date: not in use, since dimensionality charts show no existence of date variables
# State

def get_state_region(state: str) -> str:
    # Define the mapping of states to regions
    region_mapping = {
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

    # Return the region for the given state
    return region_mapping.get(state, "Unknown")


def derive_state(df: DataFrame) -> DataFrame:
    df["Region"] = df["State"].apply(get_state_region)
    return df


data_ext_state: DataFrame = derive_state(pos_covid_data)
analyse_property_granularity(data_ext_state, "State", ["Region", "State"])
plt.tight_layout()
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_state.svg")
plt.show()

# Health Days
"""property = 'HealthDays'
analyse_property_granularity(pos_covid_data, property, ["PhysicalHealthDays", 'MentalHealthDays'])
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.svg")
plt.show()"""


# Sleep Hours: aggregation: SleepHours and AtLeastEight

def is_at_least_eigh_hours(hours: float) -> bool | None:
    if hours > 24:
        return None
    elif hours >= 8:
        return True
    else:
        return False


def derive_sleep(df: DataFrame) -> DataFrame:
    df['AtLeastEightHours'] = pos_covid_data['SleepHours'].apply(is_at_least_eigh_hours)
    df.dropna(subset=['AtLeastEightHours'], inplace=True)
    return df


data_ext: DataFrame = derive_sleep(pos_covid_data)
analyse_property_granularity(data_ext, "Sleep", ["AtLeastEightHours", "SleepHours"])
plt.tight_layout()
plt.savefig(f"images/granularity/{credit_score_file_tag}_granularity_sleep.svg")
plt.show()


# Smoke Status - aggregations: SmokerStatus and NeverSmoked
def never_smoked(smoker_status: str) -> bool:
    return smoker_status == 'Never smoked'


def derive_smoker_status(df: DataFrame) -> DataFrame:
    df['NeverSmoked'] = pos_covid_data['SmokerStatus'].apply(never_smoked)
    return df


data_ext: DataFrame = derive_smoker_status(pos_covid_data)
analyse_property_granularity(data_ext, "SmokeStatus", ["NeverSmoked", "SmokerStatus"])
plt.tight_layout()
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_smoke.svg")
plt.show()


# Age - aggregations: age groups

def get_age_group(age: str) -> str:
    if (age == 'Age 18 to 24' or age == 'Age 25 to 29'):
        return 'YoungAdult'
    elif (age == 'Age 30 to 34' or age == 'Age 35 to 39' or age == 'Age 40 to 45'):
        return 'MiddleAgedAdult'
    else:
        return 'OldAgedAdult'


def is_adult(age: str) -> bool:
    if (
            age == 'Age 18 to 24' or age == 'Age 25 to 29' or age == 'Age 30 to 34' or age == 'Age 35 to 39' or age == 'Age 40 to 45'):
        return True
    else:
        return False

def derive_age(df: DataFrame) -> DataFrame:
    df['AgeGroup'] = df['AgeCategory'].apply(get_age_group)
    df['IsAdult'] = df['AgeCategory'].apply(is_adult)
    df.dropna(subset=['AgeGroup'], inplace=True)
    return df


data_ext: DataFrame = derive_age(pos_covid_data)
analyse_property_granularity(data_ext, "Age", ['IsAdult', 'AgeGroup', 'AgeCategory'])
plt.tight_layout()
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_age.svg")
plt.show()


# BMI - aggregation: BMI, classification and IsHealthy

def get_body_class(bmi: float) -> str:
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    elif bmi < 35:
        return 'Obesity'
    elif bmi > 40:
        return 'Extreme Obesity'


def get_has_healthy_body(body_class: str):
    return 'Healthy' if (body_class == 'Normal') else 'NotHealthy'


def derive_body_status(df: DataFrame) -> DataFrame:
    df['BodyClassification'] = df['BMI'].apply(get_body_class)
    df['HasHealthyBody'] = df['BodyClassification'].apply(get_has_healthy_body)
    # df.dropna(subset=['AgeGroup'], inplace=True)
    return df


data_ext: DataFrame = derive_body_status(pos_covid_data)
analyse_property_granularity(data_ext, "Body Staus", ['HasHealthyBody', 'BodyClassification', 'BMI'])
plt.tight_layout()
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_body_status.svg")
plt.show()


# Tetanus protection - aggregation: TetanusLast10Tdap and IsProtectedAgainstTetanus

def get_is_protected_against_tetanus(tetanus_last_10_t_dap: str) -> bool:
    # some records come as <class 'float'>: nan
    if not isinstance(tetanus_last_10_t_dap, str): return False
    return 'Yes' in tetanus_last_10_t_dap


def derive_tetanus_protection(df: DataFrame) -> DataFrame:
    df['TetanusProtection'] = df['TetanusLast10Tdap'].apply(get_is_protected_against_tetanus)
    return df


data_ext: DataFrame = derive_tetanus_protection(pos_covid_data)
analyse_property_granularity(data_ext, "Tetanus Protection", ['TetanusProtection', 'TetanusLast10Tdap'])
plt.tight_layout()
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_tetanus_protection.svg")
plt.show()


# ------------------
# Granularity analysis for dataset "class_credit_score"
# ------------------

# MonthlyBalance - aggregations: semester, quarter, month

def get_month_number(month: str) -> int:
    return {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }[month]


def get_month_quarter(month: str) -> str:
    if get_month_number(month) <= 3:
        return "Q1"
    elif get_month_number(month) <= 6:
        return "Q2"
    elif get_month_number(month) <= 9:
        return "Q3"
    else:
        return "Q4"


def get_month_semester(month: str) -> str:
    if get_month_number(month) <= 6:
        return "S1"
    else:
        return "S2"


def derive_month(df: DataFrame) -> DataFrame:
    df["Quarter"] = df["Month"].apply(get_month_quarter)
    df["Semester"] = df["Month"].apply(get_month_semester)
    return df


data_ext: DataFrame = derive_month(credit_score_data)
analyse_property_granularity(data_ext, "Month", ["Semester", "Quarter", "Month", ])
plt.tight_layout()
plt.savefig(f"images/granularity/{credit_score_file_tag}_granularity_month.svg")
plt.show()


# Age - aggregations: age groups

def is_age_valid(age: str) -> bool:
    if not age.isnumeric() or int(age) > 120:
        return False
    return True


def get_age_group(age: str) -> str | None:
    if not is_age_valid(age):
        return None
    age = int(age)
    if age <= 0:
        return None
    if age <= 16:
        return 'Child'
    elif age <= 30:
        return 'YoungAdult'
    elif age <= 45:
        return 'MiddleAgedAdult'
    else:
        return 'OldAgedAdult'


def is_adult(age: str) -> bool:
    if not is_age_valid(age):
        return False
    age = int(age)
    if age < 18:
        return False
    else:
        return True


def derive_age(df: DataFrame) -> DataFrame:
    df['AgeGroup'] = df['Age'].apply(get_age_group)
    df['IsAdult'] = df['Age'].apply(is_adult)
    df.dropna(subset=['AgeGroup', 'IsAdult'], inplace=True)
    return df


data_ext: DataFrame = derive_age(credit_score_data)
analyse_property_granularity(data_ext, "Age", ['IsAdult', 'AgeGroup', 'Age'])
plt.tight_layout()
plt.savefig(f"images/granularity/{credit_score_file_tag}_granularity_age.svg")
plt.show()


# Occupation - aggregations: occupation groups

def get_occupation_group(occupation: str) -> str:
    if occupation in ['Scientist', 'Engineer', 'Developer', 'Doctor']:
        return 'STEM'
    elif occupation in ['Teacher']:
        return 'Education'
    elif occupation in ['Entrepreneur', 'Manager']:
        return 'Business'
    elif occupation in ['Lawyer']:
        return 'Legal'
    elif occupation in ['Media Manager', 'Journalist']:
        return 'Media'
    elif occupation in ['Accountant']:
        return 'Finance'
    elif occupation in ['Musician', 'Writer']:
        return 'Creative'
    elif occupation in ['Architect']:
        return 'Design'
    elif occupation in ['Mechanic']:
        return 'Other'
    else:
        return 'Unknown'


def derive_occupation(df: DataFrame) -> DataFrame:
    df['OccupationGroup'] = df['Occupation'].apply(get_occupation_group)
    df.dropna(subset=['OccupationGroup'], inplace=True)
    return df


data_ext: DataFrame = derive_occupation(credit_score_data)
analyse_property_granularity(data_ext, "Occupation", ['OccupationGroup', 'Occupation'])
plt.tight_layout()
plt.savefig(f"images/granularity/{credit_score_file_tag}_granularity_occupation.svg")
