from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
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

vars: list = pos_covid_data.columns.to_list()
if [] != vars:
    target = "stroke"

    n: int = len(vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, ax=axs[i, j - 1])
    savefig(f"images/{pos_covid_file_tag}_sparsity_study.png")
    show()
else:
    print("Sparsity class: there are no variables.")


vars: list = credit_score_data.columns.to_list()
if [] != vars:
    target = "stroke"

    n: int = len(vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, ax=axs[i, j - 1])
    savefig(f"images/{credit_score_file_tag}_sparsity_study.png")
    show()
else:
    print("Sparsity class: there are no variables.")

# ------------------
# Sparsity per class analysis
# ------------------

vars: list = pos_covid_data.columns.to_list()
if [] != vars:
    target = "stroke"

    n: int = len(vars) - 1
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, target, ax=axs[i, j - 1])
    savefig(f"images/{pos_covid_file_tag}_sparsity_per_class_study.png")
    show()
else:
    print("Sparsity per class: there are no variables.")

vars: list = credit_score_data.columns.to_list()
if [] != vars:
    target = "stroke"

    n: int = len(vars) - 1
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, target, ax=axs[i, j - 1])
    savefig(f"images/{pos_covid_file_tag}_sparsity_per_class_study.png")
    show()
else:
    print("Sparsity per class: there are no variables.")

# ------------------
# Correlation analysis
# ------------------

from seaborn import heatmap
from utils.dslabs_functions import get_variable_types

variables_types: dict[str, list] = get_variable_types(pos_covid_data)
numeric: list[str] = variables_types["numeric"]
corr_mtx: DataFrame = pos_covid_data[numeric].corr().abs()

figure()
heatmap(
    abs(corr_mtx),
    xticklabels=numeric,
    yticklabels=numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/{pos_covid_file_tag}_correlation_analysis.png")
show()

variables_types: dict[str, list] = get_variable_types(credit_score_data)
numeric: list[str] = variables_types["numeric"]
corr_mtx: DataFrame = credit_score_data[numeric].corr().abs()

figure()
heatmap(
    abs(corr_mtx),
    xticklabels=numeric,
    yticklabels=numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/{pos_covid_file_tag}_correlation_analysis.png")
show()
