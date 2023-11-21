import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap

from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart
from utils.dslabs_functions import get_variable_types

run_credit_score_sparsity_analysis: bool = True
run_credit_score_sparsity_per_class_analysis: bool = True
run_credit_score_correlation_analysis: bool = True
run_sampling: bool = False
sampling_amount: float = 0.01

credit_score_filename = "../../data/credit_score/class_credit_score.csv"
credit_score_savefig_path_prefix = "images/sparsity/class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")

if run_sampling:
    credit_score_data = credit_score_data.sample(frac=sampling_amount, random_state=42)

credit_score_data = credit_score_data.dropna()
credit_score_vars: list = credit_score_data.columns.to_list()

# ------------------
# Sparsity analysis
# ------------------

print("Printing sparsity analysis for credit score...")
if credit_score_vars and run_credit_score_sparsity_analysis:
    n: int = len(credit_score_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        print(f"i: {i}")
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, ax=axs[i, j - 1])
    print("Tightening layout...")
    plt.tight_layout()
    print("Saving image for credit score sparsity study...")
    plt.savefig(f"{credit_score_savefig_path_prefix}_sparsity_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not credit_score_vars:
        print("Sparsity class: there are no variables.")
    if not run_credit_score_sparsity_analysis:
        print("Sparsity analysis: skipping.")

# ------------------
# Sparsity per class analysis
# ------------------

print("Printing sparsity per class analysis for credit score...")
if credit_score_vars and run_credit_score_sparsity_per_class_analysis:
    target = "Credit_Score"

    n: int = len(credit_score_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, target, ax=axs[i, j - 1])
    print("Saving image for credit score sparsity per class study...")
    plt.savefig(f"{credit_score_savefig_path_prefix}_sparsity_per_class_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not credit_score_vars:
        print("Sparsity per class: there are no variables.")
    if not run_credit_score_sparsity_per_class_analysis:
        print("Sparsity per class analysis: skipping.")

# ------------------
# Correlation analysis
# ------------------

print("Printing correlation analysis for credit score...")
if credit_score_vars and run_credit_score_correlation_analysis:
    # TODO: However, to our knowledge, there isn't a pre-existing method to compute the correlation between symbolic variables, nor between symbolic and numeric ones. The easiest way to deal with this situation is then to convert all symbolic variables to numeric ones, and then to compute the correlation matrix for the dataset. (See how to do it in the Variable Encoding lab, in the Data Preparation chapter).

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
    plt.savefig(f"{credit_score_savefig_path_prefix}_correlation_analysis.png")
    plt.show()
    plt.clf()
else:
    if not credit_score_vars:
        print("Correlation analysis: there are no variables.")
    if not run_credit_score_correlation_analysis:
        print("Correlation analysis: skipping.")
