import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import analyse_property_granularity, plot_bar_chart, HEIGHT

pos_covid_filename = "../../data/class_pos_covid.csv"
pos_covid_savefig_path_prefix = "images/granularity/class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values=[''])


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
vars = ['Region', 'State']
cols: int = len(vars)
fig: plt.Figure
fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
fig.suptitle(f"Granularity study for Region")
for i in range(cols):
    counts = data_ext_state[vars[i]].value_counts()
    vertical_bar_label = True if (i == 1) else False
    plot_bar_chart(
        counts.index.to_list(),
        counts.to_list(),
        ax=axs[0, i],
        title=vars[i],
        xlabel=vars[i],
        ylabel="nr records",
        percentage=False
    )
    if i == 1:
        axs[0, i].set_xticks(counts.index.to_list(), labels=counts.index.to_list(), rotation=90)
plt.tight_layout()
plt.savefig(f"{pos_covid_savefig_path_prefix}_granularity_region.png")
plt.show()

# Health Days
"""property = 'HealthDays'
analyse_property_granularity(pos_covid_data, property, ["PhysicalHealthDays", 'MentalHealthDays'])
plt.savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
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
    return df


data_ext_state: DataFrame = derive_sleep(pos_covid_data)
vars = ['AtLeastEightHours', 'SleepHours']
cols: int = len(vars)
fig: plt.Figure
fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
fig.suptitle(f"Granularity study for Sleep")
for i in range(cols):
    counts = data_ext_state[vars[i]].value_counts()
    vertical_bar_label = True if (i == 1) else False
    plot_bar_chart(
        counts.index.to_list(),
        counts.to_list(),
        ax=axs[0, i],
        title=vars[i],
        xlabel=vars[i],
        ylabel="nr records",
        percentage=False
    )
    if i == 1:
        axs[0, i].set_xticks(counts.index.to_list(), labels=counts.index.to_list(), rotation=90)
plt.tight_layout()
plt.savefig(f"{pos_covid_savefig_path_prefix}_granularity_sleep.png")
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
plt.savefig(f"{pos_covid_savefig_path_prefix}_granularity_smoke.png")
plt.show()


# Age - aggregations: age groups

def get_age_group(age: str) -> str:
    if age == 'Age 18 to 24' or age == 'Age 25 to 29':
        return 'YoungAdult'
    elif age == 'Age 30 to 34' or age == 'Age 35 to 39' or age == 'Age 40 to 45':
        return 'MiddleAgedAdult'
    else:
        return 'OldAgedAdult'


def is_adult(age: str) -> bool:
    if age == 'Age 18 to 24' or age == 'Age 25 to 29' or age == 'Age 30 to 34' or age == 'Age 35 to 39' or age == 'Age 40 to 45':
        return True
    else:
        return False


def derive_age(df: DataFrame) -> DataFrame:
    df['AgeGroup'] = df['AgeCategory'].apply(get_age_group)
    df['IsAdult'] = df['AgeCategory'].apply(is_adult)
    return df


data_ext: DataFrame = derive_age(pos_covid_data)
analyse_property_granularity(data_ext, "Age", ['IsAdult', 'AgeGroup', 'AgeCategory'])
plt.tight_layout()
plt.savefig(f"{pos_covid_savefig_path_prefix}_granularity_age.png")
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
    return df


data_ext: DataFrame = derive_body_status(pos_covid_data)
analyse_property_granularity(data_ext, "Body Status", ['HasHealthyBody', 'BodyClassification', 'BMI'])
plt.tight_layout()
plt.savefig(f"{pos_covid_savefig_path_prefix}_granularity_body_status.png")
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

vars = ['TetanusProtection', 'TetanusLast10Tdap']
cols: int = len(vars)
fig: plt.Figure
fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
fig.suptitle(f"Granularity study for Tetanus Protection")
for i in range(cols):
    counts = data_ext[vars[i]].value_counts()
    plot_bar_chart(
        counts.index.to_list(),
        counts.to_list(),
        ax=axs[0, i],
        title=vars[i],
        xlabel=vars[i],
        ylabel="nr records",
        percentage=False,
    )
    if i == 1:
        axs[0, i].set_xticks(counts.index.to_list(), labels=counts.index.to_list(), rotation=90)
plt.tight_layout()
plt.savefig(f"{pos_covid_savefig_path_prefix}_granularity_tetanus_protection.png")
plt.show()
