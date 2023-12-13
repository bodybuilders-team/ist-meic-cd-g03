import matplotlib.pyplot as plt
from numpy import std, argsort
from sklearn.ensemble import RandomForestClassifier

from utils.dslabs_functions import (
    CLASS_EVAL_METRICS,
    read_train_test_from_files, random_forests_study, plot_horizontal_bar_chart,
)
from utils.dslabs_functions import plot_evaluation_results, plot_multiline_chart

train_filename = "../../data/pos_covid/processed_data/class_pos_covid_train_lowvar.csv"
test_filename = "../../data/pos_covid/processed_data/class_pos_covid_test_lowvar.csv"
pos_covid_file_tag: str = "class_pos_covid"
target = "CovidPos"

run_sampling = True
sampling_amount = 0.01 if run_sampling else 1

sample_tag = f"_1_{int(1 / sampling_amount)}th" if run_sampling else ""

run_parameter_study = True
run_performance_analysis = True
run_variable_importance = True
run_overfitting_study = True

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target,
                                                                  sampling_amount)
print(f'Train#={len(trnX)} Test#={len(tstX)}')
print(f'Labels={labels}')

# ----------------------------
# Parameter Study
# ----------------------------

eval_metric = "accuracy"

if run_parameter_study:
    plt.figure()
    best_model, params = random_forests_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_trees=1000,
        lag=250,
        metric=eval_metric,
    )

    plt.tight_layout()
    plt.savefig(f"images/{pos_covid_file_tag}_rf_{eval_metric}_study{sample_tag}.png")
    plt.show()
    plt.clf()

# Best alternative: d=5, f=0.7

# ----------------------------
# Performance Analysis
# ----------------------------

if run_performance_analysis:
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plt.figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=pos_covid_file_tag, sample_tag=sample_tag)
    plt.tight_layout()
    plt.show()
    plt.clf()

# ----------------------------
# Variable Importance
# ----------------------------

if run_variable_importance:
    stdevs: list[float] = list(std([tree.feature_importances_ for tree in best_model.estimators_], axis=0))
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f + 1}. {elems[f]} ({importances[indices[f]]})")

    plt.figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="RF variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    plt.tight_layout()
    plt.savefig(f"images/{pos_covid_file_tag}_rf_{eval_metric}_vars_ranking{sample_tag}.png")
    plt.show()
    plt.clf()

# ----------------------------
# Overfitting Study
# ----------------------------

if run_overfitting_study:
    d_max: int = params["params"][0]
    feat: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric: str = "accuracy"

    for n in nr_estimators:
        clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
        clf.fit(trnX, trnY)
        prd_tst_Y = clf.predict(tstX)
        prd_trn_Y = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    plt.figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"RF overfitting study for d={d_max} and f={feat}",
        xlabel="nr_estimators",
        ylabel=str(eval_metric),
        percentage=True,
    )
    plt.savefig(f"images/{pos_covid_file_tag}_rf_{eval_metric}_overfitting{sample_tag}.png")
    plt.show()
    plt.clf()
