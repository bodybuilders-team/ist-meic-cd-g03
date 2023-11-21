import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from pandas import Series
from pandas import read_csv, DataFrame

from utils.dslabs_functions import get_variable_types, define_grid, HEIGHT, histogram_with_distributions, \
    set_chart_labels, plot_multibar_chart, count_outliers
from utils.dslabs_functions import plot_bar_chart

pos_covid_filename = "../../data/class_pos_covid.csv"
pos_covid_savefig_path_prefix = "images/distribution/class_pos_covid"

pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
pos_covid_summary = pos_covid_data.describe()

pos_covid_variables_types: dict[str, list] = get_variable_types(pos_covid_data)
pos_covid_numeric: list[str] = pos_covid_variables_types["numeric"]

run_pos_covid_global_boxplot: bool = True
run_pos_covid_single_boxplots: bool = True
run_pos_covid_outliers: bool = True
run_pos_covid_histograms: bool = True
run_pos_covid_histograms_with_distributions: bool = True
run_pos_covid_histograms_symbolic: bool = True
run_pos_covid_class_distribution: bool = True

# ------------------
# Global boxplots
# ------------------

if pos_covid_numeric and run_pos_covid_global_boxplot:
    print("Printing global boxplot for pos_covid...")

    pos_covid_data[pos_covid_numeric].boxplot(rot=45)
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_global_boxplot.png")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_numeric:
        print("There are no numeric variables.")
    if not run_pos_covid_global_boxplot:
        print("Global boxplots: skipping analysis.")

# ------------------
# Single variable boxplots
# ------------------

if pos_covid_numeric and run_pos_covid_single_boxplots:
    print("Printing single variable boxplots for pos_covid...")

    rows, cols = define_grid(len(pos_covid_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(pos_covid_numeric)):
        axs[i, j].set_title("Boxplot for %s" % pos_covid_numeric[n])
        axs[i, j].boxplot(pos_covid_data[pos_covid_numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_single_boxplots.png")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_numeric:
        print("There are no numeric variables.")
    if not run_pos_covid_single_boxplots:
        print("Single variable boxplots: skipping analysis.")

# ------------------
# Outliers
# ------------------

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

if pos_covid_numeric and run_pos_covid_outliers:
    print("Printing outliers for pos_covid...")

    outliers: dict[str, int] = count_outliers(pos_covid_data, pos_covid_numeric)
    plt.figure(figsize=(12, HEIGHT))
    plt.tight_layout()
    plot_multibar_chart(
        pos_covid_numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    plt.savefig(f"{pos_covid_savefig_path_prefix}_outliers_standard.png")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_numeric:
        print("There are no numeric variables.")
    if not run_pos_covid_outliers:
        print("Outliers: skipping analysis.")

# ------------------
# Histograms
# ------------------

if pos_covid_numeric and run_pos_covid_histograms:
    print("Printing histograms for pos_covid...")

    rows, cols = define_grid(len(pos_covid_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(pos_covid_numeric)):
        data = pos_covid_data[pos_covid_numeric[n]].dropna().values
        print(f"{pos_covid_numeric[n]} - {len(data)}")
        # Print 5 random samples of the data
        # samples = np.random.choice(data, size=5, replace=False)
        # print(samples)
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {pos_covid_numeric[n]}",
            xlabel=pos_covid_numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_single_histograms_numeric.png")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_numeric:
        print("There are no numeric variables.")
    if not run_pos_covid_histograms:
        print("Histograms: skipping analysis.")

# ------------------
# Distributions
# ------------------

if pos_covid_numeric and run_pos_covid_histograms_with_distributions:
    print("Printing histograms with distributions for pos_covid...")

    rows, cols = define_grid(len(pos_covid_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(pos_covid_numeric)):
        data = pos_covid_data[pos_covid_numeric[n]].dropna()
        print(f"{pos_covid_numeric[n]} - {len(data)}")
        histogram_with_distributions(axs[i, j], data, pos_covid_numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_histogram_numeric_distribution.png")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_numeric:
        print("There are no numeric variables.")
    if not run_pos_covid_histograms_with_distributions:
        print("Histograms with distributions: skipping analysis.")

# ------------------
# Histograms for Symbolic Variables
# ------------------

pos_covid_symbolic: list[str] = pos_covid_variables_types["symbolic"] + pos_covid_variables_types["binary"]
if pos_covid_symbolic and run_pos_covid_histograms_symbolic:
    print("Printing histograms for symbolic variables for pos_covid...")

    rows, cols = define_grid(len(pos_covid_symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(pos_covid_symbolic)):
        counts: Series = pos_covid_data[pos_covid_symbolic[n]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % pos_covid_symbolic[n],
            xlabel=pos_covid_symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_histograms_symbolic.png")
    # plt.show()
    plt.clf()
else:
    if not pos_covid_symbolic:
        print("There are no symbolic variables.")
    if not run_pos_covid_histograms_symbolic:
        print("Histograms for symbolic variables: skipping analysis.")

# ------------------
# Class Distribution
# ------------------

if run_pos_covid_class_distribution:
    print("Printing class distribution for pos_covid...")

    target = "CovidPos"

    values: Series = pos_covid_data[target].value_counts()

    plt.figure(figsize=(4, 2))
    plot_bar_chart(
        values.index.to_list(),
        values.to_list(),
        title=f"Target distribution (target={target})",
    )
    plt.tight_layout()
    plt.savefig(f"{pos_covid_savefig_path_prefix}_class_distribution.png")
    # plt.show()
    plt.clf()
else:
    print("Class Distribution: skipping analysis.")
