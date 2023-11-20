from pandas import read_csv, concat, DataFrame, Series
from matplotlib.pyplot import figure, show
from utils.dslabs_functions import plot_bar_chart

"""
A dataset is unbalanced if the number of samples in each class is not similar - this can bias the model.
"""

pos_covid_filename: str = "../../data/class_pos_covid.csv"
pos_covid_file_tag: str = "class_pos_covid"
pos_covid_data: DataFrame = read_csv(pos_covid_filename, na_values="")
target: str = "CovidPos"

target_count: Series = pos_covid_data[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)
values: dict[str, list] = {
    "Original": [target_count[positive_class], target_count[negative_class]]
}

figure()
plot_bar_chart(
    target_count.index.to_list(), target_count.to_list(), title="Class balance"
)
show()

# TODO: Analyze the results of the chart above

# ------------------
# Approach 1: Undersampling
# ------------------



# ------------------
# Approach 2: Oversampling
# ------------------

...

# ------------------
# Approach 3: SMOTE
# ------------------

...