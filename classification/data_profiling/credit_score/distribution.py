import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from pandas import Series
from pandas import read_csv, DataFrame

from utils.dslabs_functions import get_variable_types, define_grid, HEIGHT, plot_multibar_chart, set_chart_labels
from utils.dslabs_functions import plot_bar_chart, count_outliers, histogram_with_distributions

credit_score_filename = "../../data/class_credit_score.csv"
credit_score_savefig_path_prefix = "images/distribution/class_credit_score"

credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")
credit_score_summary = credit_score_data.describe()

credit_score_variables_types: dict[str, list] = get_variable_types(credit_score_data)
credit_score_numeric: list[str] = credit_score_variables_types["numeric"]

run_credit_score_global_boxplot: bool = True
run_credit_score_single_boxplots: bool = True
run_credit_score_outliers: bool = True
run_credit_score_histograms: bool = True
run_credit_score_histograms_with_distributions: bool = True
run_credit_score_histograms_symbolic: bool = True
run_credit_score_class_distribution: bool = True

# ------------------
# Global boxplots
# ------------------

# if credit_score_numeric:
#     frame = credit_score_data[credit_score_numeric]
#     frame = frame[frame > -10e15]
#     frame = frame[frame < 10e15]
#
#     ax = frame.boxplot(rot=45)
#     ax.set_yscale('symlog')
#
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.savefig(f"{credit_score_savefig_path_prefix}_global_boxplot.png")
#     plt.show()
#     plt.clf()
# else:
#     print("There are no numeric variables.")

# TODO: Fix this, should show a plot like above

if credit_score_numeric and run_credit_score_global_boxplot:
    print("Printing global boxplot for credit score...")

    frame = credit_score_data[credit_score_numeric]

    ax = frame.boxplot(rot=45)
    ax.set_yscale('symlog')

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_global_boxplot.png")
    # plt.show()
    plt.clf()
else:
    if not credit_score_numeric:
        print("There are no numeric variables.")
    if not run_credit_score_global_boxplot:
        print("Global boxplots: skipping analysis.")

# ------------------
# Single variable boxplots
# ------------------


if credit_score_numeric and run_credit_score_single_boxplots:
    print("Printing single variable boxplots for credit score...")

    rows, cols = define_grid(len(credit_score_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(credit_score_numeric)):
        axs[i, j].set_yscale('symlog')
        axs[i, j].set_title("Boxplot for %s" % credit_score_numeric[n])
        axs[i, j].boxplot(credit_score_data[credit_score_numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_single_boxplots.png")
    # plt.show()
    plt.clf()
else:
    if not credit_score_numeric:
        print("There are no numeric variables.")
    if not run_credit_score_single_boxplots:
        print("Single variable boxplots: skipping analysis.")

# ------------------
# Outliers
# ------------------

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

if credit_score_numeric and run_credit_score_outliers:
    print("Printing outliers for credit score...")

    outliers: dict[str, int] = count_outliers(credit_score_data, credit_score_numeric)
    plt.figure(figsize=(12, HEIGHT))
    plt.tight_layout()
    plot_multibar_chart(
        credit_score_numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    plt.xticks(rotation=90)
    plt.savefig(f"{credit_score_savefig_path_prefix}_outliers_standard.png")
    # plt.show()
    plt.clf()
else:
    if not credit_score_numeric:
        print("There are no numeric variables.")
    if not run_credit_score_outliers:
        print("Outliers: skipping analysis.")

# ------------------
# Histograms
# ------------------


if credit_score_numeric and run_credit_score_histograms:
    print("Printing histograms for credit score...")

    rows, cols = define_grid(len(credit_score_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(credit_score_numeric)):
        data = credit_score_data[credit_score_numeric[n]].dropna().values
        print(f"{credit_score_numeric[n]} - {len(data)}")
        # Print 5 random samples of the data
        # samples = np.random.choice(data, size=5, replace=False)
        # print(samples)
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {credit_score_numeric[n]}",
            xlabel=credit_score_numeric[n],
            ylabel="nr records",
        )
        if credit_score_numeric[n] == "MonthlyBalance":
            axs[i, j].hist(data, bins=1000)
        else:
            axs[i, j].hist(data, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_single_histograms_numeric.png")
    # plt.show()
    plt.clf()
else:
    if not credit_score_numeric:
        print("There are no numeric variables.")
    if not run_credit_score_histograms:
        print("Histograms: skipping analysis.")

# ------------------
# Distributions
# ------------------


if credit_score_numeric and run_credit_score_histograms_with_distributions:
    print("Printing histograms with distributions for credit_score...")

    rows, cols = define_grid(len(credit_score_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(credit_score_numeric)):
        data = credit_score_data[credit_score_numeric[n]].dropna()
        print(f"{credit_score_numeric[n]} - {len(data)}")
        histogram_with_distributions(axs[i, j], data, credit_score_numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_histogram_numeric_distribution.png")
    # plt.show()
    plt.clf()
else:
    if not credit_score_numeric:
        print("There are no numeric variables.")
    if not run_credit_score_histograms_with_distributions:
        print("Histograms with distributions: skipping analysis.")

# ------------------
# Histograms for Symbolic Variables
# ------------------

credit_score_symbolic: list[str] = credit_score_variables_types["symbolic"] + credit_score_variables_types["binary"]
if credit_score_symbolic and run_credit_score_histograms_symbolic:
    print("Printing histograms for symbolic variables credit_score...")

    rows, cols = define_grid(len(credit_score_symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(credit_score_symbolic)):
        counts: Series = credit_score_data[credit_score_symbolic[n]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % credit_score_symbolic[n],
            xlabel=credit_score_symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_histograms_symbolic.png")
    # plt.show()
    plt.clf()
else:
    if not credit_score_symbolic:
        print("There are no symbolic variables.")
    if not run_credit_score_histograms_symbolic:
        print("Histograms for symbolic variables: skipping analysis.")

# ------------------
# Class Distribution
# ------------------

if run_credit_score_class_distribution:
    print("Printing class distribution for credit_score...")

    target = "Credit_Score"

    values: Series = credit_score_data[target].value_counts()

    plt.figure(figsize=(4, 2))
    plot_bar_chart(
        values.index.to_list(),
        values.to_list(),
        title=f"Target distribution (target={target})",
    )
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_class_distribution.png")
    # plt.show()
    plt.clf()
else:
    print("Class Distribution: skipping analysis.")
