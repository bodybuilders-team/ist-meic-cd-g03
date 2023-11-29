from typing import Literal

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils.dslabs_functions import read_train_test_from_files, CLASS_EVAL_METRICS, knn_study, HEIGHT, \
    plot_evaluation_results, plot_multiline_chart

train_filename = "../../data/credit_score/processed_data/class_credit_score_train_under.csv"
test_filename = "../../data/credit_score/processed_data/class_credit_score_test.csv"
credit_score_file_tag: str = "class_credit_score"
target = "Credit_Score"

run_sampling = True
sampling_amount = 0.01 if run_sampling else 1

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target,
                                                                  sampling_amount)
print(f'Train#={len(trnX)} Test#={len(tstX)}')
print(f'Labels={labels}')

# ----------------------------
# Parameter Study
# ----------------------------

plt.figure()
eval_metrics = list(CLASS_EVAL_METRICS.keys())
cols = len(eval_metrics)
fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
fig.suptitle(f"KNN Study ({credit_score_file_tag})")
for i in range(len(eval_metrics)):
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=25, metric=eval_metrics[i], ax=axs[0][i])

fig.tight_layout()
fig.savefig(f"images/{credit_score_file_tag}_knn_study.png")
fig.show()

# Best alternative: Manhattan with k=25
best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=25, metric="accuracy")
print(f"Best model: {best_model}")
plt.clf()

# ----------------------------
# Performance Analysis
# ----------------------------

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plt.figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=credit_score_file_tag)
plt.tight_layout()
plt.show()
plt.clf()

# ----------------------------
# Overfitting Study
# ----------------------------

distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
K_MAX = 25
kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
y_tst_values: list = []
y_trn_values: list = []
acc_metric: str = "accuracy"
for k in kvalues:
    clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
    clf.fit(trnX, trnY)
    prd_tst_Y = clf.predict(tstX)
    prd_trn_Y = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

plt.figure()
plot_multiline_chart(
    kvalues,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"KNN overfitting study for {distance}",
    xlabel="K",
    ylabel=str(acc_metric),
    percentage=True,
)
plt.tight_layout()
plt.savefig(f"images/{credit_score_file_tag}_knn_overfitting.png")
plt.show()
plt.clf()