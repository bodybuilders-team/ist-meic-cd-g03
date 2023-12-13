import matplotlib.pyplot as plt
from numpy import argsort, std
from sklearn.ensemble import GradientBoostingClassifier

from utils.dslabs_functions import read_train_test_from_files, gradient_boosting_study, plot_evaluation_results, \
    plot_horizontal_bar_chart, plot_multiline_chart, CLASS_EVAL_METRICS

train_filename = "../../data/pos_covid/processed_data/class_pos_covid_train_lowvar.csv"
test_filename = "../../data/pos_covid/processed_data/class_pos_covid_test_lowvar.csv"
pos_covid_file_tag: str = "class_pos_covid"
target = "CovidPos"

run_sampling = False
sampling_amount = 0.01 if run_sampling else 1

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target,
                                                                  sampling_amount)
print(f'Train#={len(trnX)} Test#={len(tstX)}')
print(f'Labels={labels}')

# ----------------------------
# Parameter Study
# ----------------------------

eval_metric = "accuracy"

plt.figure()
best_model, params = gradient_boosting_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=1000,
    lag=250,
    metric=eval_metric,
)
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_gb_{eval_metric}_study.png")
plt.show()
plt.clf()

# Best alternative: 100 trees (d=2 and lr=0.1)

# ----------------------------
# Performance Analysis
# ----------------------------

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plt.figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=pos_covid_file_tag)
plt.tight_layout()
plt.show()
plt.clf()

# ----------------------------
# Variable Importance
# ----------------------------

trees_importances: list[float] = []
for lst_trees in best_model.estimators_:
    for tree in lst_trees:
        trees_importances.append(tree.feature_importances_)

stdevs: list[float] = list(std(trees_importances, axis=0))
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
    title="GB variables importance",
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_gb_{eval_metric}_vars_ranking.png")
plt.show()
plt.clf()

# ----------------------------
# Overfitting Analysis
# ----------------------------

d_max: int = params["params"][0]
lr: float = params["params"][1]
nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric: str = "accuracy"

for n in nr_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
    clf.fit(trnX, trnY)
    prd_tst_Y = clf.predict(tstX)
    prd_trn_Y = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

plt.figure()
plot_multiline_chart(
    nr_estimators,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"GB overfitting study for d={d_max} and lr={lr}",
    xlabel="nr_estimators",
    ylabel=str(eval_metric),
    percentage=True,
)
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_gb_{eval_metric}_overfitting.png")
plt.show()
plt.clf()
