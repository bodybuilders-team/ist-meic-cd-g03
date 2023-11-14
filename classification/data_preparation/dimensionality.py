from pandas import read_csv, DataFrame

filename = "data/class_pos_covid.csv"
file_tag = "class_pos_covid"
data: DataFrame = read_csv(filename, na_values="", index_col="id")

# TODO: Change and finish
