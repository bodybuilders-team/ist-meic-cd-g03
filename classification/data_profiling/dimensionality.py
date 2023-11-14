from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from utils.chart_utils import plot_bar_chart
from utils.dslabs_functions import get_variable_types

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")

# ------------------
# Ration between Nr. of records and Nr. of variables
# ------------------

print(f"Pos Covid Data: {pos_covid_data.shape[0]} records, {pos_covid_data.shape[1]} variables")
print(f"Credit Score Data: {credit_score_data.shape[0]} records, {credit_score_data.shape[1]} variables")

figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": pos_covid_data.shape[0], "nr variables": pos_covid_data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"images/{pos_covid_file_tag}_records_variables.png")
show()

figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": credit_score_data.shape[0], "nr variables": credit_score_data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"images/{credit_score_file_tag}_records_variables.png")
show()

# ------------------
# Nr. of variables per type
# ------------------

pos_covid_variable_types: dict[str, list] = get_variable_types(pos_covid_data)
print(f"Pos Covid Data: {len(pos_covid_variable_types)} variable types")

counts: dict[str, int] = {}
for tp in pos_covid_variable_types.keys():
    counts[tp] = len(pos_covid_variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{pos_covid_file_tag}_variable_types.png")
show()

credit_score_variable_types: dict[str, list] = get_variable_types(credit_score_data)
print(f"Credit Score Data: {len(credit_score_variable_types)} variable types")
counts: dict[str, int] = {}
for tp in credit_score_variable_types.keys():
    counts[tp] = len(credit_score_variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{credit_score_file_tag}_variable_types.png")
show()

# ------------------
# Nr. missing values per variable
# ------------------

pos_covid_mv: dict[str, int] = {}
for var in pos_covid_data.columns:
    nr: int = pos_covid_data[var].isna().sum()
    if nr > 0:
        pos_covid_mv[var] = nr

figure()
plot_bar_chart(
    list(pos_covid_mv.keys()),
    list(pos_covid_mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"images/{pos_covid_file_tag}_mv.png")
show()

credit_score_mv: dict[str, int] = {}
for var in credit_score_data.columns:
    nr: int = credit_score_data[var].isna().sum()
    if nr > 0:
        credit_score_mv[var] = nr

figure()
plot_bar_chart(
    list(credit_score_mv.keys()),
    list(credit_score_mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"images/{credit_score_file_tag}_mv.png")
show()