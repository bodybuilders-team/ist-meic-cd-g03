from numpy import ndarray
from pandas import DataFrame

from utils.dslabs_functions import CLASS_EVAL_METRICS, run_NB, run_KNN


def evaluate_approach(train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy") -> dict[
    str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval


"""
Example of test:

target = "stroke"
file_tag = "stroke"
train: DataFrame = read_csv("data/stroke_train.csv")
test: DataFrame = read_csv("data/stroke_test.csv")

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval.png")
show()"""
