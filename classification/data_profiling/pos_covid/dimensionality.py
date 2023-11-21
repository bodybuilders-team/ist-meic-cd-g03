import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

from utils.dslabs_functions import get_variable_types, plot_bar_chart

pos_covid_filename: str = "../../data/class_pos_covid.csv"
pos_covid_savefig_path_prefix = "images/dimensionality/class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")

run_pos_covid_records_variables_analysis: bool = True
run_pos_covid_variable_types_analysis: bool = True
run_pos_covid_missing_values_analysis: bool = True

# ------------------
# Ration between Nr. of records and Nr. of variables
# ------------------

if run_pos_covid_records_variables_analysis:
    print(f"Pos Covid Data: {pos_covid_data.shape[0]} records, {pos_covid_data.shape[1]} variables")

    plt.figure()
    pos_covid_values: dict[str, int] = {"nr records": pos_covid_data.shape[0],
                                        "nr variables": pos_covid_data.shape[1]}
    plot_bar_chart(
        list(pos_covid_values.keys()), list(pos_covid_values.values()), title="Nr of records vs nr variables"
    )
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_records_variables.png")
    plt.show()
    plt.clf()
else:
    print("Ration between Nr. of records and Nr. of variables: skipping analysis.")

# ------------------
# Nr. of variables per type
# ------------------

if run_pos_covid_variable_types_analysis:
    pos_covid_variable_types: dict[str, list] = get_variable_types(pos_covid_data)
    print(f"Pos Covid Data: {len(pos_covid_variable_types)} variable types")

    counts: dict[str, int] = {}
    for tp in pos_covid_variable_types.keys():
        counts[tp] = len(pos_covid_variable_types[tp])

    plt.figure()
    plot_bar_chart(
        list(counts.keys()), list(counts.values()), title="Nr of variables per type"
    )
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_variable_types.png")
    plt.show()
    plt.clf()
else:
    print("Nr. of variables per type: skipping analysis.")

# ------------------
# Nr. missing values per variable
# ------------------

if run_pos_covid_missing_values_analysis:
    pos_covid_mv: dict[str, int] = {}
    for var in pos_covid_data.columns:
        nr: int = pos_covid_data[var].isna().sum()
        if nr > 0:
            pos_covid_mv[var] = nr

    plt.figure(figsize=(8, 4))
    plot_bar_chart(
        list(pos_covid_mv.keys()),
        list(pos_covid_mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values",
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_mv.png")
    plt.show()
    plt.clf()
else:
    print("Nr. missing values per variable: skipping analysis.")
