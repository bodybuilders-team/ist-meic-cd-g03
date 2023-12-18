from typing import Literal

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils.dslabs_functions import read_train_test_from_files, CLASS_EVAL_METRICS, plot_evaluation_results, \
    plot_multiline_chart, knn_study2

train_filename = "../../data/credit_score/processed_data/class_credit_score_train_lowvar.csv"
test_filename = "../../data/credit_score/processed_data/class_credit_score_test_lowvar.csv"
credit_score_file_tag: str = "class_credit_score"
target = "Credit_Score"

run_sampling = False
sampling_amount = 0.1 if run_sampling else 1

sample_tag = f"_1_{int(1 / sampling_amount)}th" if run_sampling else ""

run_parameter_study = True
run_overfitting_study = True

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target,
                                                                  sampling_amount)
print(f'Train#={len(trnX)} Test#={len(tstX)}')
print(f'Labels={labels}')

# ----------------------------
# Parameter Study
# ----------------------------

if run_parameter_study:
    best_model, params = knn_study2(trnX, trnY, tstX, tstY, k_max=25, file_tag=credit_score_file_tag)
    plt.tight_layout()
    plt.savefig(f"images/{credit_score_file_tag}_knn_study{sample_tag}.png")
    plt.show()
    print(f"Best model: {best_model}")
    plt.clf()

    # Performance Analysis

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plt.figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=credit_score_file_tag,
                            sample_tag=sample_tag)
    plt.tight_layout()
    plt.show()
    plt.clf()

# ----------------------------
# Overfitting Study
# ----------------------------

if run_overfitting_study:
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
    plt.savefig(f"images/{credit_score_file_tag}_knn_overfitting{sample_tag}.png")
    plt.show()
    plt.clf()
