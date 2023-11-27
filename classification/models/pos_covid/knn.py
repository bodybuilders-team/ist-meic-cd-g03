'''
    Use this script to test a given change in each data-preparation pipeline step.
    This script uses Naive Bayes classifier.
'''

import matplotlib.pyplot as plt

from utils.dslabs_functions import knn_study, split_train_test_from_file

pos_covid_train_filename: str = "../../data/pos_covid/processed_data/class_pos_covid_train_smote.csv"
pos_covid_test_filename = "../../data/pos_covid/processed_data/class_pos_covid_test.csv"

pos_covid_file_tag: str = "class_pos_covid"
eval_metric = "accuracy"


def test_approach(filename, save_filename, target):
    trnX, tstX, trnY, tstY, labels = split_train_test_from_file(filename, target)

    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")

    plt.figure()
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=19, metric=eval_metric)
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.show()

    # prd_trn: array = best_model.predict(trnX)
    # prd_tst: array = best_model.predict(tstX)
    # plt.figure()
    # plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    # plt.savefig(f'images/{}_{params["name"]}_best_{params["metric"]}_eval.png')
    # plt.show()


def test_covid_approach(train_filename, save_filename):
    test_approach(train_filename, save_filename, target="CovidPos")


test_covid_approach("../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach1.csv",
                    f"images/{pos_covid_file_tag}_imputed_mv_approach1_knn_{eval_metric}_study.png")

test_covid_approach("../../data/pos_covid/processed_data/class_pos_covid_imputed_mv_approach2.csv",
                    f"images/{pos_covid_file_tag}_imputed_mv_approach2_knn_{eval_metric}_study.png")
