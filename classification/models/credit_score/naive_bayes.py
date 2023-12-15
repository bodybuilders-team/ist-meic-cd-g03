import matplotlib.pyplot as plt

from utils.dslabs_functions import read_train_test_from_files, naive_Bayes_study, CLASS_EVAL_METRICS, HEIGHT, \
    plot_evaluation_results

train_filename = "../../data/credit_score/processed_data/class_credit_score_train_lowvar.csv"
test_filename = "../../data/credit_score/processed_data/class_credit_score_test_lowvar.csv"
credit_score_file_tag: str = "class_credit_score"
target = "Credit_Score"

run_sampling = False
sampling_amount = 0.01 if run_sampling else 1

sample_tag = f"_1_{int(1 / sampling_amount)}th" if run_sampling else ""

run_nb_alternatives_study = True
run_performance_analysis = True

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target,
                                                                  sampling_amount)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

# ----------------------------
# Study Naive Bayes Alternatives
# ----------------------------

if run_nb_alternatives_study:
    plt.figure()
    eval_metrics = list(CLASS_EVAL_METRICS.keys())
    cols = len(eval_metrics)
    fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Naive Bayes Study ({credit_score_file_tag})")
    for i in range(len(eval_metrics)):
        best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metrics[i], ax=axs[0][i])

    fig.tight_layout()
    fig.savefig(f"images/{credit_score_file_tag}_nb_study{sample_tag}.png")
    fig.show()

# Best alternative: GaussianNB (better in all metrics)
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "accuracy")
print(f"Best model: {params["name"]}")
plt.clf()

# ----------------------------
# Performance Analysis
# ----------------------------

if run_performance_analysis:
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plt.figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=credit_score_file_tag, sample_tag=sample_tag)
    plt.tight_layout()
    plt.show()
    plt.clf()
