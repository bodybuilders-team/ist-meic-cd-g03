'''
file:       dslabs_functions.py
version:    2023.1
'''
from datetime import datetime
from itertools import product
from math import pi, sin, cos, ceil
from numbers import Number

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import gca, gcf, savefig, subplots
from numpy import arange, ndarray, set_printoptions, array
from numpy import log
from pandas import DataFrame, read_csv, concat, unique, to_numeric, to_datetime, Series, Index
from scipy.stats import norm, expon, lognorm
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

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


def plot_bar_chart(xvalues: list, yvalues: list, ax: Axes = None, title: str = '', xlabel: str = '',
                   ylabel: str = '', percentage: bool = False):
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax = set_chart_xticks(xvalues, ax=ax, percentage=percentage)
    values = ax.bar(xvalues, yvalues, label=yvalues, edgecolor=LINE_COLOR, color=FILL_COLOR, tick_label=xvalues)
    format = '%.2f' if percentage else '%.0f'
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


# ---------------------------------------
#             DATA PREPARATION
# ---------------------------------------

def derive_date_variables(df, date_vars):
    for date in date_vars:
        df[date + '_year'] = df[date].dt.year
        df[date + '_quarter'] = df[date].dt.quarter
        df[date + '_month'] = df[date].dt.month
        df[date + '_day'] = df[date].dt.day
    return df


def encode_cyclic_variables(data, vars):
    for v in vars:
        x_max = max(data[v])
        data[v + '_sin'] = data[v].apply(lambda x: round(sin(2 * pi * x / x_max), 3))
        data[v + '_cos'] = data[v].apply(lambda x: round(cos(2 * pi * x / x_max), 3))


def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=bool, drop='if_binary')
    trans = enc.fit_transform(df[vars_to_dummify])
    print(trans)

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


def mvi_by_filling(data: DataFrame, strategy: str = 'frequent') -> DataFrame:
    '''
        data: DataFrame - the data to clean
        strategy: str - the strategy to apply ('frequent', 'constant' or 'knn')
        return the data modified
    '''
    df, tmp_nr, tmp_sb, tmp_bool = None, None, None, None
    variables = get_variable_types(data)

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
    return df


# ---------------------------------------
#             CLASSIFICATION
# ---------------------------------------

DELTA_IMPROVE: float = 0.001

CLASS_EVAL_METRICS = {
    'accuracy': accuracy_score,
    'recall': recall_score,
    'precision': precision_score,
    'auc': roc_auc_score,
    'f1': f1_score,
}


def read_train_test_from_files(train_fn: str, test_fn: str, target: str = 'class'):
    train = read_csv(train_fn, index_col=None)
    trnY = train.pop(target).values
    trnX = train.values
    labels = unique(trnY)
    labels.sort()

    test = read_csv(test_fn, index_col=None)
    tstY = test.pop(target).values
    tstX = test.values
    return trnX, tstX, trnY, tstY, labels, train.columns


def split_train_test_from_file(fn: str, target: str = 'class'):
    df = read_csv(fn, index_col=None)
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
    savefig(f'images/{file_tag}_{model['name']}_best_{model['metric']}_eval.png')
    return axs


def naive_Bayes_study(trnX, trnY, tstX, tstY, metric='accuracy', file_tag=''):
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

    plot_bar_chart(xvalues, yvalues, title=f'Naive Bayes Models ({metric})', ylabel=metric, percentage=True)
    savefig(f'images/{file_tag}_nb_{metric}_study.png')

    return best_model, best_params


def knn_study(trnX, trnY, tstX, tstY, k_max=19, lag=2, metric='accuracy', file_tag=''):
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

    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)
    savefig(f'images/{file_tag}_knn_{metric}_study.png')

    return best_model, best_params


def evaluate_approach(train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy") -> dict[
    str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = knn_study(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval


def select_low_variance_variables(data: DataFrame, max_threshold: float, target: str = "class") -> list:
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
    for thresh in options:
        vars2drop: Index[str] = summary5.columns[
            summary5.loc["std"] * summary5.loc["std"] < thresh
            ]
        vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop

        train_copy: DataFrame = train.drop(vars2drop, axis=1, inplace=False)
        test_copy: DataFrame = test.drop(vars2drop, axis=1, inplace=False)
        eval: dict[str, list] | None = evaluate_approach(
            train_copy, test_copy, target=target, metric=metric
        )
        if eval is not None:
            results["NB"].append(eval[metric][0])
            results["KNN"].append(eval[metric][1])

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


def select_redundant_variables(data: DataFrame, min_threshold: float = 0.90, target: str = "class") -> list:
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
    variables: Index[str] = corr_matrix.columns
    results: dict[str, list] = {"NB": [], "KNN": []}
    for thresh in options:
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


def run_NB(trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy") -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)
        # print(f'NB {clf}')
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


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


def set_chart_labels(
        ax: Axes, title: str = "", xlabel: str = "", ylabel: str = ""
) -> Axes:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def set_chart_xticks(
        xvalues: list[str | int | float | datetime], ax: Axes, percentage: bool = False
) -> Axes:
    if len(xvalues) > 0:
        if percentage:
            ax.set_ylim(0.0, 1.0)

        if isinstance(xvalues[0], datetime):
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(
                AutoDateFormatter(locator, defaultfmt="%Y-%m-%d")
            )
        rotation: int = 0
        if not any(not isinstance(x, (int, float)) for x in xvalues):
            ax.set_xlim(left=xvalues[0], right=xvalues[-1])
            ax.set_xticks(xvalues, labels=xvalues)
        else:
            rotation = 45

        ax.tick_params(axis="x", labelrotation=rotation, labelsize="xx-small")

    return ax


def plot_bar_chart(
        xvalues: list,
        yvalues: list,
        ax: Axes = None,  # type: ignore
        title: str = "",
        xlabel:
        str = "",
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
