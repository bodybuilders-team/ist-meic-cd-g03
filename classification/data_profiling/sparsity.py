from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from numpy import ndarray
from pandas import read_csv, DataFrame

from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
pos_covid_data = pos_covid_data.dropna()

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")
credit_score_data = credit_score_data.dropna()

# ------------------
# Sparsity analysis
# ------------------

pos_covid_vars: list = pos_covid_data.columns.to_list()
if [] != pos_covid_vars:
    target = "stroke"

    n: int = len(pos_covid_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(pos_covid_vars)):
        var1: str = pos_covid_vars[i]
        for j in range(i + 1, len(pos_covid_vars)):
            var2: str = pos_covid_vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, ax=axs[i, j - 1])
    savefig(f"images/sparsity/{pos_covid_file_tag}_sparsity_study.png")
    show()
else:
    print("Sparsity class: there are no variables.")


credit_score_vars: list = credit_score_data.columns.to_list()
if [] != credit_score_vars:
    target = "stroke"

    n: int = len(credit_score_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, ax=axs[i, j - 1])
    savefig(f"images/sparsity/{credit_score_file_tag}_sparsity_study.png")
    show()
else:
    print("Sparsity class: there are no variables.")

# ------------------
# Sparsity per class analysis
# ------------------

pos_covid_vars: list = pos_covid_data.columns.to_list()
if [] != pos_covid_vars:
    target = "stroke"

    n: int = len(pos_covid_vars) - 1
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(pos_covid_vars)):
        var1: str = pos_covid_vars[i]
        for j in range(i + 1, len(pos_covid_vars)):
            var2: str = pos_covid_vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, target, ax=axs[i, j - 1])
    savefig(f"images/sparsity/{pos_covid_file_tag}_sparsity_per_class_study.png")
    show()
else:
    print("Sparsity per class: there are no variables.")

credit_score_vars: list = credit_score_data.columns.to_list()
if [] != credit_score_vars:
    target = "stroke"

    n: int = len(credit_score_vars) - 1
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, target, ax=axs[i, j - 1])
    savefig(f"images/sparsity/{credit_score_file_tag}_sparsity_per_class_study.png")
    show()
else:
    print("Sparsity per class: there are no variables.")

# ------------------
# Correlation analysis
# ------------------

from seaborn import heatmap
from utils.dslabs_functions import get_variable_types

pos_covid_variables_types: dict[str, list] = get_variable_types(pos_covid_data)
pos_covid_numeric: list[str] = pos_covid_variables_types["numeric"]
pos_covid_corr_mtx: DataFrame = pos_covid_data[pos_covid_numeric].corr().abs()

figure()
heatmap(
    abs(pos_covid_corr_mtx),
    xticklabels=pos_covid_numeric,
    yticklabels=pos_covid_numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/sparsity/{pos_covid_file_tag}_correlation_analysis.png")
show()

credit_score_variables_types: dict[str, list] = get_variable_types(credit_score_data)
credit_score_numeric: list[str] = credit_score_variables_types["numeric"]
credit_score_corr_mtx: DataFrame = credit_score_data[credit_score_numeric].corr().abs()

figure()
heatmap(
    abs(credit_score_corr_mtx),
    xticklabels=credit_score_numeric,
    yticklabels=credit_score_numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/sparsity/{credit_score_file_tag}_correlation_analysis.png")
show()
