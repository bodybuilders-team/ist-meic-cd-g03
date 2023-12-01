'''
file:       dslabs_functions.py
version:    2023.1
'''
from datetime import datetime
from itertools import product
from math import pi, sin, cos, ceil
from numbers import Number
from typing import Callable, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import gca, gcf, savefig, subplots, show, tight_layout
from numpy import arange, ndarray, set_printoptions, array
from numpy import log
from pandas import DataFrame, read_csv, concat, unique, to_numeric, to_datetime, Series, Index, Period
from scipy.stats import norm, expon, lognorm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from utils.config import ACTIVE_COLORS, LINE_COLOR, FILL_COLOR, cmap_blues

NR_COLUMNS: int = 3
HEIGHT: int = 4

TEXT_MARGIN = 0.05
FONT_SIZE = 6
FONT_TEXT = FontProperties(size=FONT_SIZE)

alpha = 0.3

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5


# _reset_epoch_test_example()
# set_epoch('0000-12-31T00:00:00')  # old epoch (pre MPL 3.3)

# ---------------------------------------
#             DATA CHARTS
# ---------------------------------------

def define_grid(nr_vars, vars_per_row: int = NR_COLUMNS) -> tuple:
    nr_rows = 1
    if nr_vars % vars_per_row == 0:
        nr_rows = nr_vars // vars_per_row
    else:
        nr_rows = nr_vars // vars_per_row + 1
    return nr_rows, vars_per_row


def set_chart_labels(ax, title: str = '', xlabel: str = '', ylabel: str = ''):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def set_chart_xticks(xvalues: list, ax: Axes, percentage: bool = False):
    if len(xvalues) > 0:
        if percentage:
            ax.set_ylim(0.0, 1.0)

        if isinstance(xvalues[0], datetime):
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(AutoDateFormatter(locator, defaultfmt='%Y-%m-%d'))

        rotation = 0
        if isinstance(xvalues[0], Number):
            ax.set_xlim((xvalues[0], xvalues[-1]))
            ax.set_xticks(xvalues, labels=xvalues)
        else:
            rotation = 45

        ax.tick_params(axis='x', labelrotation=rotation, labelsize='xx-small')

    return ax


def plot_line_chart(xvalues: list, yvalues: list, ax: Axes = None, title: str = '', xlabel: str = '',
                    ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax, percentage=percentage)
    ax.plot(xvalues, yvalues, c=LINE_COLOR)
    return ax


def plot_bar_chart(
        xvalues: list,
        yvalues: list,
        ax: Axes = None,  # type: ignore
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        percentage: bool = False,
) -> Axes:
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    values: BarContainer = ax.bar(
        xvalues,
        yvalues,
        label=yvalues,
        edgecolor=LINE_COLOR,
        color=FILL_COLOR,
        tick_label=xvalues,
    )
    format = "%.2f" if percentage else "%.0f"
    ax.bar_label(values, fmt=format, fontproperties=FONT_TEXT)

    return ax


def plot_scatter_chart(var1: list, var2: list, ax: Axes = None, title: str = '', xlabel: str = '', ylabel: str = ''):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.scatter(var1, var2)
    return ax


def plot_multiline_chart(xvalues: list, yvalues: dict, ax: Axes = None, title: str = '', xlabel: str = '',
                         ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    legend: list = []
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend)
    return ax


def plot_multibar_chart(group_labels: list, yvalues: dict, ax: Axes = None, title: str = '',
                        xlabel: str = '', ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    bar_labels = list(yvalues.keys())

    # This is the location for each bar
    index = arange(len(group_labels))
    bar_width = 0.8 / len(bar_labels)
    ax.set_xticks(index + bar_width / 2, labels=group_labels)

    for i in range(len(bar_labels)):
        values = ax.bar(index + i * bar_width, yvalues[bar_labels[i]], width=bar_width, label=bar_labels[i])
        format = '%.2f' if percentage else '%.0f'
        ax.bar_label(values, fmt=format, fontproperties=FONT_TEXT)
    ax.legend(fontsize=FONT_SIZE)
    return ax


def plot_multi_scatters_chart(data: DataFrame, var1: str, var2: str, var3: str = None, ax: Axes = None):
    if ax is None:
        ax = gca()

    title = f'{var1} x {var2}'
    if var3 is not None:
        title += f' per {var3}'
        values = data[var3].unique().tolist()
        if len(values) > 2:
            chart = ax.scatter(data[var1], data[var2], c=data[var3])
            cbar = gcf().colorbar(chart)
            cbar.outline.set_visible(False)
            cbar.set_label(var3, loc='top')
        else:
            values.sort()
            for i in range(len(values)):
                subset = data[data[var3] == values[i]]
                ax.scatter(subset[var1], subset[var2], color=ACTIVE_COLORS[i], label=values[i])
            ax.legend(fontsize='xx-small')
    else:
        ax.scatter(data[var1], data[var2], color=FILL_COLOR)
    ax = set_chart_labels(ax=ax, title=title, xlabel=var1, ylabel=var2)
    return ax


def plot_horizontal_bar_chart(elements: list, values: list, error: list = None, ax: Axes = None, title: str = '',
                              xlabel: str = '', ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = gca()
    if percentage:
        ax.set_xlim((0, 1))
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    y_pos = arange(len(elements))

    ax.barh(y_pos, values, xerr=error, align='center', error_kw={'lw': 0.5, 'ecolor': 'r'})
    ax.set_yticks(y_pos, labels=elements)
    ax.invert_yaxis()  # labels read top-to-bottom
    return ax


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


# ---------------------------------------
#             DATA PROFILING
# ---------------------------------------

def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {'numeric': [], 'binary': [], 'date': [], 'symbolic': []}

    nr_values = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types['binary'].append(c)
            df[c].astype('bool')
        else:
            try:
                to_numeric(df[c], errors='raise')
                variable_types['numeric'].append(c)
            except:
                try:
                    df[c] = to_datetime(df[c], errors='raise')
                    variable_types['date'].append(c)
                except:
                    variable_types['symbolic'].append(c)

    return variable_types


def determine_outlier_thresholds_for_var(summary5, std_based: bool = True, threshold: int = NR_STDEV):
    top, bottom = 0, 0
    if std_based:
        std = threshold * summary5['std']
        top = summary5['mean'] + std
        bottom = summary5['mean'] - std
    else:
        iqr = threshold * (summary5['75%'] - summary5['25%'])
        top = summary5['75%'] + iqr
        bottom = summary5['25%'] - iqr

    return top, bottom


def count_outliers(data: DataFrame, numeric, nrstdev: int = NR_STDEV, iqrfactor: float = IQR_FACTOR):
    outliers_iqr = []
    outliers_stdev = []
    summary5 = data[numeric].describe()

    for var in numeric:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=True, threshold=nrstdev)
        outliers_stdev += [data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]]

        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=iqrfactor)
        outliers_iqr += [data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]]

    return {'iqr': outliers_iqr, 'stdev': outliers_stdev}


def analyse_date_granularity(data, var, levels, file_tag=''):
    cols = len(levels)
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f'Granularity study for {var}')

    for i in range(cols):
        counts = data[var + '_' + levels[i]].value_counts()
        plot_bar_chart(counts.index.to_list(), counts.values, ax=axs[0][i], title=levels[i], xlabel=levels[i],
                       ylabel='nr records', percentage=False)
    savefig(f'images/{file_tag}_granularity_{var}.png')


def analyse_property_granularity(data: DataFrame, property: str, vars: list[str]) -> ndarray:
    cols: int = len(vars)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {property}")
    for i in range(cols):
        counts: Series[int] = data[vars[i]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="nr records",
            percentage=False,
        )
    return axs


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


# ---------------------------------------
#             DATA PREPARATION
# ---------------------------------------


def encode_cyclic_variables(data, vars):
    for v in vars:
        x_max = max(data[v])
        data[v + '_sin'] = data[v].apply(lambda x: round(sin(2 * pi * x / x_max), 3))
        data[v + '_cos'] = data[v].apply(lambda x: round(cos(2 * pi * x / x_max), 3))


def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    other_vars: list[str] = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=bool, drop='if_binary')
    trans = enc.fit_transform(df[vars_to_dummify])

    new_vars = enc.get_feature_names_out(vars_to_dummify)
    dummy = DataFrame(trans, columns=new_vars, index=df.index)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df


def mvi_by_dropping(data: DataFrame, min_pct_per_variable: float = 0.1, min_pct_per_record: float = 0.0) -> DataFrame:
    '''
        data: DataFrame - the data to clean
        min_pct_per_variable: float - the minimum percentage of records a variable has to show in order to be kept
        min_pct_per_record: float - the minimum percentage of values that a record has to show in order to be kept
        return the data modified
    '''
    # Deleting variables
    df = data.dropna(axis=1, thresh=data.shape[0] * min_pct_per_variable, inplace=False)
    # Deleting records
    df.dropna(axis=0, thresh=data.shape[1] * min_pct_per_record, inplace=True)

    return df


def mvi_by_filling(data: DataFrame, strategy: str = 'frequent',
                   variable_types: dict[str, list[str]] = None) -> DataFrame:
    '''
        data: DataFrame - the data to clean
        strategy: str - the strategy to apply ('frequent', 'constant' or 'knn')
        variable_types: dict - the variable types to use for each variable, list of 'binary' and list of 'categorical'
        return the data modified
    '''
    df, tmp_nr, tmp_sb, tmp_bool = None, None, None, None
    variables = get_variable_types(data)

    # If variable types are not provided, infer them to be 'numeric', 'symbolic' or 'binary'
    if variable_types is None:
        stg_num, v_num = 'mean', -1
        stg_sym, v_sym = 'most_frequent', 'NA'
        stg_bool, v_bool = 'most_frequent', False
        if strategy != 'knn':
            if strategy == 'constant':
                stg_num, stg_sym, stg_bool = 'constant', 'constant', 'constant'
            if len(variables['numeric']) > 0:
                imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
                tmp_nr = DataFrame(imp.fit_transform(data[variables['numeric']]), columns=variables['numeric'])
            if len(variables['symbolic']) > 0:
                imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
                tmp_sb = DataFrame(imp.fit_transform(data[variables['symbolic']]), columns=variables['symbolic'])
            if len(variables['binary']) > 0:
                imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
                tmp_bool = DataFrame(imp.fit_transform(data[variables['binary']]), columns=variables['binary'])

            df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
        else:
            imp = KNNImputer(n_neighbors=5)
            imp.fit(data)
            ar = imp.transform(data)
            df = DataFrame(ar, columns=data.columns, index=data.index)
    else:
        # If variable types are provided, use them to fill the missing values
        true_numeric = [c for c in variables['numeric'] if c not in variable_types['categorical']]
        all_binary = list(set(variable_types['binary'] + variables['binary']))

        stg_num, v_num = 'mean', -1
        stg_sym, v_sym = 'most_frequent', 'NA'
        stg_bool, v_bool = 'most_frequent', False

        if strategy != 'knn':
            if strategy == 'constant':
                stg_num, stg_sym, stg_bool = 'constant', 'constant', 'constant'
            if len(variables['numeric']) > 0:
                imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
                tmp_nr = DataFrame(imp.fit_transform(data[true_numeric]),
                                   columns=true_numeric)
            if len(variable_types['categorical']) > 0:
                imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
                tmp_sb = DataFrame(imp.fit_transform(data[variable_types['categorical']]),
                                   columns=variable_types['categorical'])
            if len(variable_types['binary']) > 0:
                imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
                tmp_bool = DataFrame(imp.fit_transform(data[all_binary]),
                                     columns=all_binary)

            df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
        else:
            imp = KNNImputer(n_neighbors=5)
            imp.fit(data)
            ar = imp.transform(data)
            df = DataFrame(ar, columns=data.columns, index=data.index)

    return df


# ---------------------------------------
#             CLASSIFICATION
# ---------------------------------------

DELTA_IMPROVE: float = 0.001

CLASS_EVAL_METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "auc": roc_auc_score,
    "f1": f1_score,
}


def run_NB(trnX, trnY, tstX, tstY, metric: str = "accuracy") -> dict[str, float]:
    estimators: dict[str, GaussianNB | MultinomialNB | BernoulliNB] = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }
    best_model: GaussianNB | MultinomialNB | BernoulliNB = None  # type: ignore
    best_performance: float = 0.0
    eval: dict[str, float] = {}

    for clf in estimators:
        estimators[clf].fit(trnX, trnY)
        prdY: ndarray = estimators[clf].predict(tstX)
        performance: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if performance - best_performance > DELTA_IMPROVE:
            best_performance = performance
            best_model = estimators[clf]
    if best_model is not None:
        prd: ndarray = best_model.predict(tstX)
        for key in CLASS_EVAL_METRICS:
            eval[key] = CLASS_EVAL_METRICS[key](tstY, prd)
    return eval


def select_low_variance_variables(
        data: DataFrame, max_threshold: float, target: str = "class"
) -> list:
    summary5: DataFrame = data.describe()
    vars2drop: Index[str] = summary5.columns[
        summary5.loc["std"] * summary5.loc["std"] < max_threshold
        ]
    vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop
    return list(vars2drop.values)


def study_variance_for_feature_selection(
        train: DataFrame,
        test: DataFrame,
        target: str = "class",
        max_threshold: float = 1,
        lag: float = 0.05,
        metric: str = "accuracy",
        file_tag: str = "",
) -> dict:
    options: list[float] = [
        round(i * lag, 3) for i in range(1, ceil(max_threshold / lag + lag))
    ]
    results: dict[str, list] = {"NB": [], "KNN": []}
    summary5: DataFrame = train.describe()
    print(summary5)
    for i, thresh in enumerate(options):
        print(f"{i}/{len(options)-1}")

        vars2drop: Index[str] = summary5.columns[
            summary5.loc["std"] * summary5.loc["std"] < thresh
            ]
        vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        if len(train_copy.columns) <= 1 or len(test_copy.columns) <= 1:
            results["NB"].append(0)
            results["KNN"].append(0)
            continue

        eval: dict[str, list] | None = evaluate_approach(
            train_copy, test_copy, target=target, metric=metric
        )

        if eval is not None and len(eval) != 0:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])
        else:
            results["NB"].append(0)
            results["KNN"].append(0)

    plot_multiline_chart(
        options,
        results,
        title=f"{file_tag} variance study ({metric})",
        xlabel="variance threshold",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"images/{file_tag}_fs_low_var_{metric}_study.png")
    return results


def select_redundant_variables(
        data: DataFrame, min_threshold: float = 0.90, target: str = "class"
) -> list:
    df: DataFrame = data.drop(target, axis=1, inplace=False)
    corr_matrix: DataFrame = abs(df.corr())
    variables: Index[str] = corr_matrix.columns
    vars2drop: list = []
    for v1 in variables:
        vars_corr: Series = (corr_matrix[v1]).loc[corr_matrix[v1] >= min_threshold]
        vars_corr.drop(v1, inplace=True)
        if len(vars_corr) > 1:
            lst_corr = list(vars_corr.index)
            for v2 in lst_corr:
                if v2 not in vars2drop:
                    vars2drop.append(v2)
    return vars2drop


def study_redundancy_for_feature_selection(
        train: DataFrame,
        test: DataFrame,
        target: str = "class",
        min_threshold: float = 0.90,
        lag: float = 0.05,
        metric: str = "accuracy",
        file_tag: str = "",
) -> dict:
    options: list[float] = [
        round(min_threshold + i * lag, 3)
        for i in range(ceil((1 - min_threshold) / lag) + 1)
    ]

    df: DataFrame = train.drop(target, axis=1, inplace=False)
    corr_matrix: DataFrame = abs(df.corr())
    print(corr_matrix)
    variables: Index[str] = corr_matrix.columns
    results: dict[str, list] = {"NB": [], "KNN": []}
    for i,thresh in enumerate(options):
        print(f"{i}/{len(options)-1}")
        vars2drop: list = []
        for v1 in variables:
            vars_corr: Series = (corr_matrix[v1]).loc[corr_matrix[v1] >= thresh]
            vars_corr.drop(v1, inplace=True)
            if len(vars_corr) > 1:
                lst_corr = list(vars_corr.index)
                for v2 in lst_corr:
                    if v2 not in vars2drop:
                        vars2drop.append(v2)

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        if len(train_copy.columns) <= 1:
            results["NB"].append(0)
            results["KNN"].append(0)
            continue

        eval: dict | None = evaluate_approach(
            train_copy, test_copy, target=target, metric=metric
        )
        if eval is not None:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])

    plot_multiline_chart(
        options,
        results,
        title=f"{file_tag} redundancy study ({metric})",
        xlabel="correlation threshold",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"images/{file_tag}_fs_redundancy_{metric}_study.png")
    return results


def apply_feature_selection(
        train: DataFrame,
        test: DataFrame,
        vars2drop: list,
        filename: str = "",
        tag: str = "",
) -> tuple[DataFrame, DataFrame]:
    train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
    train_copy.to_csv(f"{filename}_train_{tag}.csv", index=True)
    test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
    test_copy.to_csv(f"{filename}_test_{tag}.csv", index=True)
    return train_copy, test_copy


def run_KNN(trnX, trnY, tstX, tstY, metric="accuracy") -> dict[str, float]:
    kvalues: list[int] = [1] + [i for i in range(5, 26, 5)]
    best_model: KNeighborsClassifier = None  # type: ignore
    best_performance: float = 0
    eval: dict[str, float] = {}
    for k in kvalues:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(trnX, trnY)
        prdY: ndarray = clf.predict(tstX)
        performance: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if performance - best_performance > DELTA_IMPROVE:
            best_performance = performance
            best_model: KNeighborsClassifier = clf
    if best_model is not None:
        prd: ndarray = best_model.predict(tstX)
        for key in CLASS_EVAL_METRICS:
            eval[key] = CLASS_EVAL_METRICS[key](tstY, prd)
    return eval


# ---------------------------------------
#             CLASSIFICATION
# ---------------------------------------


DELTA_IMPROVE: float = 0.001

CLASS_EVAL_METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "auc": roc_auc_score,
    "f1": f1_score,
}


def run_NB(trnX, trnY, tstX, tstY, metric: str = "accuracy") -> dict[str, float]:
    estimators: dict[str, GaussianNB | MultinomialNB | BernoulliNB] = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }
    best_model: GaussianNB | MultinomialNB | BernoulliNB = None  # type: ignore
    best_performance: float = 0.0
    eval: dict[str, float] = {}

    for clf in estimators:
        estimators[clf].fit(trnX, trnY)
        prdY: ndarray = estimators[clf].predict(tstX)
        performance: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if performance - best_performance > DELTA_IMPROVE:
            best_performance = performance
            best_model = estimators[clf]
    if best_model is not None:
        prd: ndarray = best_model.predict(tstX)
        for key in CLASS_EVAL_METRICS:
            eval[key] = CLASS_EVAL_METRICS[key](tstY, prd)
    return eval


def run_KNN(trnX, trnY, tstX, tstY, metric="accuracy") -> dict[str, float]:
    kvalues: list[int] = [1] + [i for i in range(5, 26, 5)]
    best_model: KNeighborsClassifier = None  # type: ignore
    best_performance: float = 0
    eval: dict[str, float] = {}
    for k in kvalues:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(trnX, trnY)
        prdY: ndarray = clf.predict(tstX)
        performance: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if performance - best_performance > DELTA_IMPROVE:
            best_performance = performance
            best_model: KNeighborsClassifier = clf
    if best_model is not None:
        prd: ndarray = best_model.predict(tstX)
        for key in CLASS_EVAL_METRICS:
            eval[key] = CLASS_EVAL_METRICS[key](tstY, prd)
    return eval


def evaluate_approach(
        train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] | None = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] | None = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval


def evaluate_approach2(trnX, trnY, tstX, tstY, metric: str = "accuracy", knn=True, nb=True) -> dict[str, list]:
    eval: dict[str, list] = {}
    eval_NB: dict[str, float] | None = {}
    eval_KNN: dict[str, float] | None = {}

    if nb:
        eval_NB: dict[str, float] | None = run_NB(trnX, trnY, tstX, tstY, metric=metric)

    if knn:
        eval_KNN: dict[str, float] | None = run_KNN(trnX, trnY, tstX, tstY, metric=metric)

    if nb and knn and eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    else:
        if eval_NB != {} and nb:
            for met in CLASS_EVAL_METRICS:
                eval[met] = [eval_NB[met], 0]
        if eval_KNN != {} and knn:
            for met in CLASS_EVAL_METRICS:
                eval[met] = [0, eval_KNN[met]]

    return eval


def evaluate_approaches(approaches: list[list], target: str = "class", study_title='', file_tag='',
                        metric: str = "accuracy", sample_amount: float = 1.0, knn=True, nb=True) -> dict[str, list]:
    cols = len(approaches)
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(study_title, fontsize=FONT_SIZE)

    for approach in approaches:
        trnX, tstX, trnY, tstY, labels = split_train_test_from_file(approach[0], target=target,
                                                                    sample_amount=sample_amount)
        eval: dict[str, list] = evaluate_approach2(trnX, trnY, tstX, tstY, metric=metric, knn=knn, nb=nb)
        plot_multibar_chart(
            ["NB", "KNN"], eval, title=f"{approach[1]} evaluation", percentage=True,
            ax=axs[0][approaches.index(approach)]
        )

    plt.tight_layout()
    savefig(f"images/{file_tag}_eval.png")
    show()


def read_train_test_from_files(train_fn: str, test_fn: str, target: str = "class", sample_amount: float = 1.0
) -> tuple[ndarray, ndarray, array, array, list, list]:
    train: DataFrame = read_csv(train_fn, index_col=None)
    if sample_amount < 1:
        train = train.sample(frac=sample_amount, random_state=42)
    labels: list = list(train[target].unique())
    labels.sort()
    trnY: array = train.pop(target).to_list()
    trnX: ndarray = train.values

    test: DataFrame = read_csv(test_fn, index_col=None)
    if sample_amount < 1:
        test = test.sample(frac=sample_amount, random_state=42)
    tstY: array = test.pop(target).to_list()
    tstX: ndarray = test.values
    return trnX, tstX, trnY, tstY, labels, train.columns.to_list()


def split_train_test_from_file(fn: str, target: str = 'class', sample_amount: float = 1.0):
    df = read_csv(fn, index_col=None)
    if sample_amount < 1:
        df = df.sample(frac=sample_amount, random_state=42)

    data_y = df.pop(target).values
    data_x = df.values
    labels = unique(data_y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(data_x, data_y, train_size=0.7, stratify=data_y)

    return trnX, tstX, trnY, tstY, labels


def plot_confusion_matrix(cnf_matrix: ndarray, classes_names: ndarray, ax: Axes = None):
    if ax is None:
        ax = gca()
    title = 'Confusion matrix'
    set_printoptions(precision=2)
    tick_marks = arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cnf_matrix, interpolation='nearest', cmap=cmap_blues)

    for i, j in product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.text(j, i, format(cnf_matrix[i, j], 'd'), color='y', horizontalalignment="center")
    return ax


def plot_roc_chart(tstY: ndarray, predictions: dict, ax: Axes = None, target: str = 'class'):
    if ax is None:
        ax = gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)

    ax.plot([0, 1], [0, 1], color='navy', label='random', linewidth=1, linestyle='--', marker='')
    models = list(predictions.keys())
    for i in range(len(models)):
        RocCurveDisplay.from_predictions(y_true=tstY, y_pred=predictions[models[i]], name=models[i],
                                         ax=ax, color=ACTIVE_COLORS[i], linewidth=1)
    ax.legend(loc="lower right")
    return ax


def plot_evaluation_results(model, trn_y, prd_trn, tst_y, prd_tst, labels: ndarray, file_tag=''):
    evaluation = {}
    for key in CLASS_EVAL_METRICS:
        evaluation[key] = [CLASS_EVAL_METRICS[key](trn_y, prd_trn), CLASS_EVAL_METRICS[key](tst_y, prd_tst)]

    params_st = '' if () == model['params'] else str(model['params'])
    fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    fig.suptitle(f'Best {model['metric']} for {model['name']} {params_st}')
    plot_multibar_chart(['Train', 'Test'], evaluation, ax=axs[0], percentage=True)

    cnf_mtx_tst = confusion_matrix(tst_y, prd_tst, labels=labels)
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1])
    tight_layout()
    savefig(f'images/{file_tag}_{model['name']}_best_{model['metric']}_eval.png')
    return axs


# ---------------------------------------
#             TIME SERIES
# ---------------------------------------

from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose


def plot_ts_multivariate_chart(data: DataFrame, title: str) -> list[Axes]:
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(
        data.shape[1], 1, figsize=(3 * HEIGHT, HEIGHT / 2 * data.shape[1])
    )
    fig.suptitle(title)

    for i in range(data.shape[1]):
        col: str = data.columns[i]
        auxi_ax: Axes = plot_line_chart(
            data[col].index.to_list(),
            data[col].to_list(),
            ax=axs[i],
            xlabel=data.index.name,
            ylabel=col,
        )
        auxi_ax.tick_params(axis="x", labelbottom="off")
    return axs


def plot_components(
        series: Series,
        title: str = "",
        x_label: str = "time",
        y_label: str = "",
) -> list[Axes]:
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return axs


def ts_aggregation_by(
        data: Series | DataFrame,
        gran_level: str = "D",
        agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df


def naive_Bayes_study(trnX, trnY, tstX, tstY, metric='accuracy', file_tag='', subtitle='', ax=None):
    estimators = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }

    xvalues = []
    yvalues = []
    best_model = None
    best_params = {'name': '', 'metric': metric, 'params': ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        eval = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance = eval
            best_params['name'] = clf
            best_model = estimators[clf]
        yvalues.append(eval)

    plot_bar_chart(
        xvalues,
        yvalues,
        title=f'Naive Bayes Models ({metric}) {" - " + subtitle if subtitle != "" else ""}',
        ylabel=metric,
        percentage=True,
        ax=ax
    )
    #savefig(f'images/{file_tag}_nb_{metric}_study.png')

    return best_model, best_params


# receives a list of approaches to study; each approach contains train_filename, subtitle
def data_prep_naive_bayes_study(approaches, metric='accuracy', study_title='', file_tag='', target='class',
                                sampling_amount=0.01):
    estimators = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }

    cols = len(approaches)
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f'Naive Bayes Models ({metric}) - {study_title}')

    for i in range(cols):
        trnX, tstX, trnY, tstY, labels = split_train_test_from_file(approaches[i][0], target=target,
                                                                    sample_amount=sampling_amount)
        xvalues = []
        yvalues = []
        best_model = None
        best_params = {'name': '', 'metric': metric, 'params': ()}
        best_performance = 0
        for clf in estimators:
            xvalues.append(clf)
            estimators[clf].fit(trnX, trnY)
            prdY = estimators[clf].predict(tstX)
            eval = CLASS_EVAL_METRICS[metric](tstY, prdY)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['name'] = clf
                best_model = estimators[clf]
            yvalues.append(eval)

        plot_bar_chart(
            xvalues,
            yvalues,
            ax=axs[0][i],
            title=approaches[i][1],
            ylabel=metric,
            percentage=True
        )

    plt.tight_layout()
    savefig(f'images/{file_tag}_nb_{metric}_study.png')
    plt.show()


def knn_study(trnX, trnY, tstX, tstY, k_max=19, lag=2, metric='accuracy', file_tag='', ax=None):
    dist = ['manhattan', 'euclidean', 'chebyshev']

    kvalues = [i for i in range(1, k_max + 1, lag)]
    best_model = None
    best_params = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance = 0

    values = {}
    for d in dist:
        y_tst_values = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY = clf.predict(tstX)
            eval = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (k, d)
                best_model = clf
        values[d] = y_tst_values
    print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')

    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True,
                         ax=ax)
    # savefig(f'images/{file_tag}_knn_{metric}_study.png')

    return best_model, best_params


def data_prep_knn_study(approaches, k_max=19, lag=2, metric='accuracy', study_title='', file_tag='', target='class',
                        sampling_amount=1):
    dist = ['manhattan', 'euclidean', 'chebyshev']

    cols = len(approaches)
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f'KNN Models ({metric}) - {study_title}')

    for i in range(cols):
        trnX, tstX, trnY, tstY, labels = split_train_test_from_file(approaches[i][0], target=target,
                                                                    sample_amount=sampling_amount)
        kvalues = [i for i in range(1, k_max + 1, lag)]
        best_model = None
        best_params = {'name': 'KNN', 'metric': metric, 'params': ()}
        best_performance = 0

        values = {}
        for d in dist:
            y_tst_values = []
            for k in kvalues:
                clf = KNeighborsClassifier(n_neighbors=k, metric=d)
                clf.fit(trnX, trnY)
                prdY = clf.predict(tstX)
                eval = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params['params'] = (k, d)
                    best_model = clf
            values[d] = y_tst_values
        print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')

        plot_multiline_chart(kvalues, values, ax=axs[0][i], title=approaches[i][1], xlabel='k', ylabel=metric,
                             percentage=True)

    plt.tight_layout()
    savefig(f'images/{file_tag}_knn_{metric}_study.png')
    plt.show()


def trees_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int = 10, lag: int = 2, metric='accuracy',
        ax=None
) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max + 1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (c, d)
                best_model = clf
            # print(f'DT {c} and d={d}')
        values[c] = y_tst_values
    print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
    plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True,
                         ax=ax)

    return best_model, best_params

def random_forests_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy"
) -> tuple[RandomForestClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for f in max_features:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = RandomForestClassifier(
                    n_estimators=n, max_depth=d, max_features=f
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, f, n)
                    best_model = clf
                # print(f'RF d={d} f={f} n={n}')
            values[f] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Random Forests with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True
        )
    print(
        f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
    )
    return best_model, best_params


def gradient_boosting_study(
        trnX: ndarray,
        trnY: array,
        tstX: ndarray,
        tstY: array,
        nr_max_trees: int = 2500,
        lag: int = 500,
        metric: str = "accuracy",
) -> tuple[GradientBoostingClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for lr in learning_rates:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = GradientBoostingClassifier(
                    n_estimators=n, max_depth=d, learning_rate=lr
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, lr, n)
                    best_model = clf
                # print(f'GB d={d} lr={lr} n={n}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params


def mlp_study(
        trnX: ndarray,
        trnY: array,
        tstX: ndarray,
        tstY: array,
        nr_max_iterations: int = 2500,
        lag: int = 500,
        metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params
