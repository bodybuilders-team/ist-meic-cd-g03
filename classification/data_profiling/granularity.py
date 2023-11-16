from matplotlib.pyplot import savefig, show
from pandas import read_csv, DataFrame

from utils.dslabs_functions import analyse_property_granularity

# ------------------
# Granularity analysis for dataset "class_pos_covid"
# ------------------

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")

# ------------------
# Granularity analysis: Date (not in use, since dimensionality charts show no existence of date variables)
# ------------------
'''
variables_types: dict[str, list] = get_variable_types(pos_covid_data)
data_ext: DataFrame = derive_date_variables(pos_covid_data, variables_types["date"])

for v_date in variables_types["date"]:
    analyse_date_granularity(pos_covid_data, v_date, ["year", "quarter", "month", "day"])
    savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{v_date}.png")
    show()
'''

# ------------------
# Granularity analysis: Location (maybe should not be used because [State] is the only property available)
# ------------------
'''
property = 'location'
analyse_property_granularity(pos_covid_data, property, ["State"])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()
'''

# ------------------
# Granularity analysis: Health Days
# ------------------

property = 'HealthDays'
analyse_property_granularity(pos_covid_data, property, ["PhysicalHealthDays", 'MentalHealthDays'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()

# ------------------
# Granularity analysis: Past Health Problems
# ------------------

property = 'PreviousHealthProblems'
analyse_property_granularity(pos_covid_data, property, ["HadHeartAttack", 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()

# ------------------
# Granularity analysis: Existing Difficulties
# ------------------

property = 'ExistingDifficulties'
analyse_property_granularity(pos_covid_data, property, ['BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()

# ------------------
# Granularity analysis: Smokes
# ------------------

property = 'Smokes'
analyse_property_granularity(pos_covid_data, property, ['SmokerStatus', 'ECigaretteUsage'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()

# ------------------
# Granularity analysis: Body Stats
# ------------------

property = 'BodyStats'
analyse_property_granularity(pos_covid_data, property, ['AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()

# ------------------
# Granularity analysis: Vaccination
# ------------------

property = 'Vaccination'
analyse_property_granularity(pos_covid_data, property, ['FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()

# ------------------
# Granularity analysis for dataset "class_credit_score"
# ------------------

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")

# ------------------
# Granularity analysis:
# ------------------

property = 'Vaccination'
analyse_property_granularity(pos_covid_data, property, ['FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap'])
savefig(f"images/granularity/{pos_covid_file_tag}_granularity_{property}.png")
show()