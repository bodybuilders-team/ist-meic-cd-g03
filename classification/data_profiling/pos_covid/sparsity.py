import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap

from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart
from utils.dslabs_functions import get_variable_types

pos_covid_filename = "../../data/pos_covid/class_pos_covid.csv"
pos_covid_savefig_path_prefix = "images/sparsity/class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
pos_covid_data = pos_covid_data.dropna()
pos_covid_vars: list = pos_covid_data.columns.to_list()

run_pos_covid_sparsity_analysis: bool = False
run_pos_covid_sparsity_per_class_analysis: bool = False
run_pos_covid_correlation_analysis: bool = True

# ------------------
# Correlation analysis
# ------------------

if pos_covid_vars and run_pos_covid_correlation_analysis:
    encoded_pos_covid_filename = "../../data/pos_covid/processed_data/class_pos_covid_encoded.csv"
    encoded_pos_covid_data: DataFrame = read_csv(encoded_pos_covid_filename, na_values="")

    pos_covid_corr_mtx: DataFrame = encoded_pos_covid_data.corr().abs()

    plt.figure(figsize=(10, 10))

    heatmap(
        abs(pos_covid_corr_mtx),
        xticklabels=pos_covid_corr_mtx.columns,
        yticklabels=pos_covid_corr_mtx.columns,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_correlation_analysis.png")
    plt.show()
    plt.clf()
else:
    if not pos_covid_vars:
        print("Correlation: there are no variables.")
    if not run_pos_covid_correlation_analysis:
        print("Correlation analysis: skipping.")

# ------------------
# Sparsity analysis
# ------------------

if pos_covid_vars and run_pos_covid_sparsity_analysis:
    n: int = len(pos_covid_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(pos_covid_vars)):
        var1: str = pos_covid_vars[i]
        for j in range(i + 1, len(pos_covid_vars)):
            var2: str = pos_covid_vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, ax=axs[i, j - 1])
    plt.tight_layout()
    print("Saving image for covid pos sparsity study...")
    plt.savefig(f"{pos_covid_savefig_path_prefix}_sparsity_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_vars:
        print("Sparsity class: there are no variables.")
    if not run_pos_covid_sparsity_analysis:
        print("Sparsity analysis: skipping.")

# ------------------
# Sparsity per class analysis
# ------------------

if pos_covid_vars and run_pos_covid_sparsity_per_class_analysis:
    target = "CovidPos"

    n: int = len(pos_covid_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(pos_covid_vars)):
        var1: str = pos_covid_vars[i]
        for j in range(i + 1, len(pos_covid_vars)):
            var2: str = pos_covid_vars[j]
            plot_multi_scatters_chart(pos_covid_data, var1, var2, target, ax=axs[i, j - 1])
    print("Saving image for covid pos sparsity per class study...")
    plt.savefig(f"{pos_covid_savefig_path_prefix}_sparsity_per_class_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_vars:
        print("Sparsity per class: there are no variables.")
    if not run_pos_covid_sparsity_per_class_analysis:
        print("Sparsity per class analysis: skipping.")
