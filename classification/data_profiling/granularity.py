import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import analyse_property_granularity

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")

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
analyse_property_granularity(data_ext_state, "State", ["State", "Region"])
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_granularity_state.svg")
plt.show()

# Health Days
"""property = 'HealthDays'
analyse_property_granularity(pos_covid_data, property, ["PhysicalHealthDays", 'MentalHealthDays'])
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.svg")
plt.show()"""

# ------------------
# Granularity analysis for dataset "class_credit_score"
# ------------------
# TODO: add granularity analysis for credit score dataset

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
analyse_property_granularity(data_ext, "Month", ["Month", "Quarter", "Semester"])
plt.tight_layout()
plt.savefig(f"images/{credit_score_file_tag}_granularity_month.svg")
plt.show()
