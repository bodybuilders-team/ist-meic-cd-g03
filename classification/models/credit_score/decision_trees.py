from typing import Literal

import matplotlib.pyplot as plt
from numpy import argsort
from sklearn.tree import plot_tree, DecisionTreeClassifier

from utils.dslabs_functions import trees_study, read_train_test_from_files, CLASS_EVAL_METRICS, HEIGHT, \
    plot_evaluation_results, plot_horizontal_bar_chart, plot_multiline_chart

train_filename = "../../data/credit_score/processed_data/class_credit_score_train_lowvar.csv"
test_filename = "../../data/credit_score/processed_data/class_credit_score_test_lowvar.csv"
credit_score_file_tag: str = "class_credit_score"
target = "Credit_Score"

run_sampling = False
sampling_amount = 0.01 if run_sampling else 1
sample_tag = f"_1_{int(1 / sampling_amount)}th" if run_sampling else ""

run_parameter_study = True
run_performance_analysis = True
run_variables_importance = True
run_overfitting_study = True

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target,
                                                                  sampling_amount)
print(f'Train#={len(trnX)} Test#={len(tstX)}')
print(f'Labels={labels}')

# ----------------------------
# Parameter Study
# ----------------------------

if run_parameter_study:
    plt.figure()
    eval_metrics = list(CLASS_EVAL_METRICS.keys())
    cols = len(eval_metrics)
    fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Decision Trees Study ({credit_score_file_tag})")
    for i in range(len(eval_metrics)):
        best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=25, metric=eval_metrics[i], ax=axs[0][i])

    fig.tight_layout()
    fig.savefig(f"images/{credit_score_file_tag}_dt_study{sample_tag}.png")
    fig.show()

# Best alternative: Entropy with max_depth=10
eval_metric = "accuracy"
best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=25, metric=eval_metric)
print(f"Best model: {best_model}")
plt.clf()

# ----------------------------
# Performance Analysis
# ----------------------------

if run_performance_analysis:
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plt.figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=credit_score_file_tag,
                            sample_tag=sample_tag)
    plt.show()
    plt.clf()

# ----------------------------
# Variables importance
# ----------------------------

if run_variables_importance:
    max_depth2show = 3
    st_labels: list[str] = [str(value) for value in labels]

    plt.figure(figsize=(14, 6))
    plot_tree(
        best_model,
        max_depth=max_depth2show,
        feature_names=vars,
        class_names=st_labels,
        filled=True,
        rounded=True,
        impurity=False,
        precision=2,
    )
    plt.tight_layout()
    plt.savefig(f"images/{credit_score_file_tag}_DT_{eval_metric}_best_tree{sample_tag}.png")
    plt.show()
    plt.clf()

    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f"{f + 1}. {elems[f]} ({importances[indices[f]]})")

    plt.figure(figsize=(5, 3))
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="Decision Tree variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    plt.tight_layout()
    plt.savefig(f"images/{credit_score_file_tag}_dt_{eval_metric}_vars_ranking{sample_tag}.png")
    plt.show()
    plt.clf()

# ----------------------------
# Overfitting Study
# ----------------------------

if run_overfitting_study:
    crit: Literal["entropy", "gini"] = params["params"][0]
    d_max = 25
    depths: list[int] = [i for i in range(2, d_max + 1, 1)]
    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric = "accuracy"
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
        clf.fit(trnX, trnY)
        prd_tst_Y = clf.predict(tstX)
        prd_trn_Y = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    plt.figure()
    plot_multiline_chart(
        depths,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"DT overfitting study for {crit}",
        xlabel="max_depth",
        ylabel=str(eval_metric),
        percentage=True,
    )
    plt.tight_layout()
    plt.savefig(f"images/{credit_score_file_tag}_dt_{eval_metric}_overfitting{sample_tag}.png")
    plt.show()
    plt.clf()
