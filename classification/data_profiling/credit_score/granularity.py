import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import analyse_property_granularity

credit_score_filename = "../../data/class_credit_score.csv"
credit_score_savefig_path_prefix = "images/granularity/class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values='', index_col="ID")


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
plt.savefig(f"{credit_score_savefig_path_prefix}_granularity_month.png")
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


def is_adult(age: str) -> bool | None:
    if not is_age_valid(age):
        return None
    age = int(age)
    if age < 18:
        return False
    else:
        return True


def derive_age(df: DataFrame) -> DataFrame:
    df['AgeGroup'] = df['Age'].apply(get_age_group)
    df['IsAdult'] = df['Age'].apply(is_adult)
    return df


data_ext: DataFrame = derive_age(credit_score_data)
analyse_property_granularity(data_ext, "Age", ['IsAdult', 'AgeGroup', 'Age'])
plt.tight_layout()
plt.savefig(f"{credit_score_savefig_path_prefix}_granularity_age.png")
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
    return df


data_ext: DataFrame = derive_occupation(credit_score_data)
analyse_property_granularity(data_ext, "Occupation", ['OccupationGroup', 'Occupation'])
plt.tight_layout()
plt.savefig(f"{credit_score_savefig_path_prefix}_granularity_occupation.png")
plt.show()
