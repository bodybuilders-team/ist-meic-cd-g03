from subprocess import call

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, axis
from sklearn.tree import export_graphviz

from utils.dslabs_functions import trees_study, read_train_test_from_files, CLASS_EVAL_METRICS, HEIGHT, \
    plot_evaluation_results

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
# Study Decision Trees Alternatives
# ----------------------------

plt.figure()
eval_metrics = list(CLASS_EVAL_METRICS.keys())
cols = len(eval_metrics)
fig, axs = plt.subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
fig.suptitle(f"Decision Trees Study ({pos_covid_file_tag})")
for i in range(len(eval_metrics)):
    best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=25, metric=eval_metrics[i], ax=axs[0][i])

fig.tight_layout()
fig.savefig(f"images/{pos_covid_file_tag}_dt_study.png")
fig.show()

# Best alternative: Entropy with max_depth=6
eval_metric = "accuracy"
best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=25, metric=eval_metric)
print(f"Best model: {best_model}")
plt.clf()

# ----------------------------
# Performance Analysis
# ----------------------------

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plt.figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, file_tag=pos_covid_file_tag)
plt.show()
plt.clf()

# ----------------------------
# Variables importance
# ----------------------------

tree_filename: str = f"images/{pos_covid_file_tag}_dt_{eval_metric}_best_tree"
max_depth2show = 3
st_labels: list[str] = [str(value) for value in labels]

dot_data: str = export_graphviz(
    best_model,
    out_file=tree_filename + ".dot",
    max_depth=max_depth2show,
    feature_names=vars,
    class_names=st_labels,
    filled=True,
    rounded=True,
    impurity=False,
    special_characters=True,
    precision=2,
)
# Convert to png
call(["dot", "-Tpng", tree_filename + ".dot", "-o", tree_filename + ".png", "-Gdpi=600"])

plt.figure(figsize=(14, 6))
imshow(imread(tree_filename + ".png"))
axis("off")
plt.show()
plt.clf()

# TODO: Finish this - Jesus