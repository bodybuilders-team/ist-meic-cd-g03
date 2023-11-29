import matplotlib.pyplot as plt

from utils.dslabs_functions import read_train_test_from_files, naive_Bayes_study, CLASS_EVAL_METRICS, HEIGHT, \
    plot_evaluation_results

train_filename = "../../data/pos_covid/processed_data/class_pos_covid_train_over.csv"
test_filename = "../../data/pos_covid/processed_data/class_pos_covid_test.csv"
pos_covid_file_tag: str = "class_pos_covid"
target = "CovidPos"

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

# ----------------------------
# Study Naive Bayes Alternatives
# ----------------------------

plt.figure()
eval_metrics = list(CLASS_EVAL_METRICS.keys())
cols = len(eval_metrics)
fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
fig.suptitle(f"Naive Bayes Study ({pos_covid_file_tag})")
for i in range(len(eval_metrics)):
    best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metrics[i], ax=axs[0][i])

fig.tight_layout()
fig.savefig(f"images/{pos_covid_file_tag}_nb_study.png")
fig.show()

# Best alternative: BernoulliNB
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "accuracy")
print(f"Best model: {best_model}")

# ----------------------------
# Performance Analysis
# ----------------------------

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plt.figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
plt.savefig(f'images/{pos_covid_file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
plt.show()
