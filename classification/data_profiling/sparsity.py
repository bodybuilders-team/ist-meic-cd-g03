import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap

from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart
from utils.dslabs_functions import get_variable_types

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
pos_covid_data = pos_covid_data.dropna()
pos_covid_vars: list = pos_covid_data.columns.to_list()

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")
credit_score_data = credit_score_data.dropna()
credit_score_vars: list = credit_score_data.columns.to_list()

# ------------------
# Sparsity analysis
# ------------------

if pos_covid_vars:
    n: int = len(pos_covid_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(pos_covid_vars)):
        var1: str = pos_covid_vars[i]
        for j in range(i + 1, len(pos_covid_vars)):
            var2: str = pos_covid_vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, ax=axs[i, j - 1])
    print("Saving image for covid pos sparsity study...")
    plt.savefig(f"images/sparsity/{pos_covid_file_tag}_sparsity_study.png")
    print("Image saved")
    plt.show()
    plt.clf()
else:
    print("Sparsity class: there are no variables.")

if credit_score_vars:
    n: int = len(credit_score_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, ax=axs[i, j - 1])
    print("Saving image for credit score sparsity study...")
    plt.savefig(f"images/sparsity/{credit_score_file_tag}_sparsity_study.png")
    print("Image saved")
    plt.show()
    plt.clf()
else:
    print("Sparsity class: there are no variables.")

# ------------------
# Sparsity per class analysis
# ------------------

if pos_covid_vars:
    target = "CovidPos"

    n: int = len(pos_covid_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(pos_covid_vars)):
        var1: str = pos_covid_vars[i]
        for j in range(i + 1, len(pos_covid_vars)):
            var2: str = pos_covid_vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, target, ax=axs[i, j - 1])
    print("Saving image for covid pos sparsity per class study...")
    plt.savefig(f"images/sparsity/{pos_covid_file_tag}_sparsity_per_class_study.png")
    print("Image saved")
    plt.show()
    plt.clf()
else:
    print("Sparsity per class: there are no variables.")

if credit_score_vars:
    target = "Credit_Score"

    n: int = len(credit_score_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, target, ax=axs[i, j - 1])
    print("Saving image for credit score sparsity per class study...")
    plt.savefig(f"images/sparsity/{credit_score_file_tag}_sparsity_per_class_study.png")
    print("Image saved")
    plt.show()
    plt.clf()
else:
    print("Sparsity per class: there are no variables.")

# ------------------
# Correlation analysis
# ------------------

# TODO: However, to our knowledge, there isn't a pre-existing method to compute the correlation between symbolic variables, nor between symbolic and numeric ones. The easiest way to deal with this situation is then to convert all symbolic variables to numeric ones, and then to compute the correlation matrix for the dataset. (See how to do it in the Variable Encoding lab, in the Data Preparation chapter).

pos_covid_variables_types: dict[str, list] = get_variable_types(pos_covid_data)
pos_covid_numeric: list[str] = pos_covid_variables_types["numeric"]
pos_covid_corr_mtx: DataFrame = pos_covid_data[pos_covid_numeric].corr().abs()

plt.figure()
heatmap(
    abs(pos_covid_corr_mtx),
    xticklabels=pos_covid_numeric,
    yticklabels=pos_covid_numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
plt.tight_layout()
plt.savefig(f"images/sparsity/{pos_covid_file_tag}_correlation_analysis.svg")
plt.show()
plt.clf()

credit_score_variables_types: dict[str, list] = get_variable_types(credit_score_data)
credit_score_numeric: list[str] = credit_score_variables_types["numeric"]
credit_score_corr_mtx: DataFrame = credit_score_data[credit_score_numeric].corr().abs()

plt.figure()
heatmap(
    abs(credit_score_corr_mtx),
    xticklabels=credit_score_numeric,
    yticklabels=credit_score_numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
plt.tight_layout()
plt.savefig(f"images/sparsity/{credit_score_file_tag}_correlation_analysis.svg")
plt.show()
plt.clf()
