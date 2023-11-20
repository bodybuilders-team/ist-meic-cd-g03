import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from numpy import log
from numpy import ndarray
from pandas import Series
from pandas import read_csv, DataFrame
from scipy.stats import norm, expon, lognorm

from utils.dslabs_functions import get_variable_types, define_grid, HEIGHT, plot_multibar_chart, set_chart_labels, \
    plot_multiline_chart

pos_covid_filename = "../data/class_pos_covid.csv"
pos_covid_file_tag = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
pos_covid_summary = pos_covid_data.describe()

credit_score_filename = "../data/class_credit_score.csv"
credit_score_file_tag = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")
credit_score_summary = credit_score_data.describe()

# ------------------
# Global boxplots
# ------------------

pos_covid_variables_types: dict[str, list] = get_variable_types(pos_covid_data)
pos_covid_numeric: list[str] = pos_covid_variables_types["numeric"]
if pos_covid_numeric:
    pos_covid_data[pos_covid_numeric].boxplot(rot=45)
    plt.tight_layout()
    plt.savefig(f"images/distribution/{pos_covid_file_tag}_global_boxplot.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

# credit_score_variables_types: dict[str, list] = get_variable_types(credit_score_data)
# credit_score_numeric: list[str] = credit_score_variables_types["numeric"]
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
#     plt.savefig(f"images/distribution/{credit_score_file_tag}_global_boxplot.svg")
#     plt.show()
#     plt.clf()
# else:
#     print("There are no numeric variables.")

# TODO: Fix this, should show a plot like above
credit_score_variables_types: dict[str, list] = get_variable_types(credit_score_data)
credit_score_numeric: list[str] = credit_score_variables_types["numeric"]
if credit_score_numeric:
    frame = credit_score_data[credit_score_numeric]

    ax = frame.boxplot(rot=45)
    ax.set_yscale('symlog')

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"images/distribution/{credit_score_file_tag}_global_boxplot.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

# ------------------
# Single variable boxplots
# ------------------

if pos_covid_numeric:
    rows, cols = define_grid(len(pos_covid_numeric))
    fig: plt.figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    fig.tight_layout()
    i, j = 0, 0
    for n in range(len(pos_covid_numeric)):
        axs[i, j].set_title("Boxplot for %s" % pos_covid_numeric[n])
        axs[i, j].boxplot(pos_covid_data[pos_covid_numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.savefig(f"images/distribution/{pos_covid_file_tag}_single_boxplots.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

if credit_score_numeric:
    rows: int
    cols: int
    rows, cols = define_grid(len(credit_score_numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    fig.tight_layout()
    i, j = 0, 0
    for n in range(len(credit_score_numeric)):
        axs[i, j].set_yscale('symlog')
        axs[i, j].set_title("Boxplot for %s" % credit_score_numeric[n])
        axs[i, j].boxplot(credit_score_data[credit_score_numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.savefig(f"images/distribution/{credit_score_file_tag}_single_boxplots.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

# ------------------
# Outliers
# ------------------

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5


def determine_outlier_thresholds_for_var(
        summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom


def count_outliers(
        data: DataFrame,
        numeric: list[str],
        nrstdev: int = NR_STDEV,
        iqrfactor: float = IQR_FACTOR,
) -> dict:
    outliers_iqr: list = []
    outliers_stdev: list = []
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}


if [] != pos_covid_numeric:
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
    plt.savefig(f"images/distribution/{pos_covid_file_tag}_outliers_standard.svg")
    plt.show()
else:
    print("There are no numeric variables.")

plt.clf()

if credit_score_numeric:
    outliers: dict[str, int] = count_outliers(credit_score_data, credit_score_numeric)
    plt.figure(figsize=(12, HEIGHT))

    plot_multibar_chart(
        credit_score_numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    plt.tight_layout()
    plt.savefig(f"images/distribution/{credit_score_file_tag}_outliers_standard.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

# ------------------
# Histograms
# ------------------

if pos_covid_numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    fig.tight_layout()
    i, j = 0, 0
    for n in range(len(pos_covid_numeric)):
        data = pos_covid_data[pos_covid_numeric[n]].dropna().values
        print(n)
        print(len(data))
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {pos_covid_numeric[n]}",
            xlabel=pos_covid_numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.savefig(f"images/distribution/{pos_covid_file_tag}_single_histograms_numeric.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

# if [] != credit_score_numeric:
#     fig, axs = subplots(
#         rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
#     )
#     fig.tight_layout()
#     i, j = 0, 0
#     for n in range(len(credit_score_numeric)):
#         data = credit_score_data[credit_score_numeric[n]].dropna().values
#         print(n)
#         print(len(data))
#         set_chart_labels(
#             axs[i, j],
#             title=f"Histogram for {credit_score_numeric[n]}",
#             xlabel=credit_score_numeric[n],
#             ylabel="nr records",
#         )
#         axs[i, j].hist(data, "auto")
#         i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
#     plt.savefig(f"images/distribution/{credit_score_file_tag}_single_histograms_numeric.svg")
#     plt.show()
#     plt.clf()
# else:
#     print("There are no numeric variables.")
#

# ------------------
# Distributions
# ------------------
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions["LogNor(%.1f,%.2f)" % (log(scale), sigma)] = lognorm.pdf(
        x_values, sigma, loc, scale
    )
    return distributions


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values: list = series.sort_values().to_list()
    ax.hist(values, 20, density=True)
    distributions: dict = compute_known_distributions(values)
    plot_multiline_chart(
        values,
        distributions,
        ax=ax,
        title="Best fit for %s" % var,
        xlabel=var,
        ylabel="",
    )


if pos_covid_numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    fig.tight_layout()
    i, j = 0, 0
    for n in range(len(pos_covid_numeric)):
        histogram_with_distributions(axs[i, j], pos_covid_data[pos_covid_numeric[n]].dropna(), pos_covid_numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.savefig(f"images/distribution/{pos_covid_file_tag}_histogram_numeric_distribution.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

plt.clf()
if credit_score_numeric:
    fig: plt.Figure
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    fig.tight_layout()
    i, j = 0, 0
    for n in range(len(credit_score_numeric)):
        histogram_with_distributions(axs[i, j], credit_score_data[credit_score_numeric[n]].dropna(),
                                     credit_score_numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.savefig(f"images/distribution/{credit_score_file_tag}_histogram_numeric_distribution.svg")
    plt.show()
    plt.clf()
else:
    print("There are no numeric variables.")

# TODO: Things related to Symbolic Variables