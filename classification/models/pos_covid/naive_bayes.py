'''
    Use this script to test a given change in each data-preparation pipeline step.
    This script uses Naive Bayes classifier.
'''

import matplotlib.pyplot as plt

from utils.dslabs_functions import naive_Bayes_study, read_train_test_from_files

pos_covid_train_filename: str = "../../data/pos_covid/processed_data/class_pos_covid_train_smote.csv"
pos_covid_test_filename = "../../data/pos_covid/processed_data/class_pos_covid_test.csv"
pos_covid_file_tag: str = "class_pos_covid"
target: str = "CovidPos"
eval_metric = "accuracy"

trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    pos_covid_train_filename, pos_covid_test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

plt.figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
plt.tight_layout()
plt.savefig(f"images/{pos_covid_file_tag}_nb_{eval_metric}_study.png")
plt.show()
