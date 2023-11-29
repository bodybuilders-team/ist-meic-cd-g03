from typing import Literal

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

from utils.dslabs_functions import (
    CLASS_EVAL_METRICS,
    read_train_test_from_files, mlp_study,
)
from utils.dslabs_functions import plot_evaluation_results, plot_multiline_chart

train_filename = "../../data/pos_covid/processed_data/class_pos_covid_train_over.csv"
test_filename = "../../data/pos_covid/processed_data/class_pos_covid_test.csv"
pos_covid_file_tag: str = "class_pos_covid"
target = "CovidPos"

run_sampling = True
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
best_model, params = mlp_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=1000,
    lag=250,
    metric=eval_metric,
)

plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_mlp_{eval_metric}_study.png")
plt.show()
plt.clf()

# Best alternative: MLP best for 250 iterations (lr_type=constant and lr=0.05

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
# Overfitting Study
# ----------------------------

lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
lr: float = params["params"][1]
nr_iterations: list[int] = [i for i in range(100, 1001, 100)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "accuracy"

warm_start: bool = False
for n in nr_iterations:
    clf = MLPClassifier(
        warm_start=warm_start,
        learning_rate=lr_type,
        learning_rate_init=lr,
        max_iter=n,
        activation="logistic",
        solver="sgd",
        verbose=False,
    )
    clf.fit(trnX, trnY)
    prd_tst_Y = clf.predict(tstX)
    prd_trn_Y = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
    warm_start = True

plt.figure()
plot_multiline_chart(
    nr_iterations,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
    xlabel="nr_iterations",
    ylabel=str(eval_metric),
    percentage=True,
)
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_mlp_{eval_metric}_overfitting.png")
plt.show()
plt.clf()
