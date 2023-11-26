from pandas import DataFrame, read_csv

from utils.dslabs_functions import dummify

credit_score_filename: str = "../../data/credit_score/class_credit_score.csv"
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")

# ------------------
# Ordinal Encoding: Binary and Categorical variables with order
# ------------------

# print ordinal variables and their values
for col in credit_score_data.columns:
    if credit_score_data[col].dtype == "object":
        print(col, credit_score_data[col].unique())

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
credit_mix_type_values: dict[str, int] = { "Good": 0, "Standard": 1, "Bad": 2}
payment_of_min_amount_type_values: dict[str, int] = {"No": 0, "NM": 1, "Yes": 2}
credit_score_type_values = {"Poor": 0, "Good": 1}
payment_behaviour_type_values: dict[str, int] = {
    "Low_spent_Small_value_payments": 0,
    "Low_spent_Medium_value_payments": 1,
    "Low_spent_Large_value_payments": 2,
    "High_spent_Small_value_payments": 3,
    "High_spent_Medium_value_payments": 4,
    "High_spent_Large_value_payments": 5,
}
month_type_values: dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

# TODO: Credit_History_Age?, SSN?, Name? Customer_ID?

encoding: dict[str, dict[str, int]] = {
    "CreditMix": credit_mix_type_values,
    "Payment_of_Min_Amount": payment_of_min_amount_type_values,
    "Credit_Score": credit_score_type_values,
    "Payment_Behaviour": payment_behaviour_type_values,
    "Month": month_type_values,
}
df: DataFrame = credit_score_data.replace(encoding, inplace=False)
print(df.head(5))

# ------------------
# Dummification
# ------------------

# TODO: "Type_of_Loan"
"""
Resposta da stora:
The issue about the Type_of_Loan variable is that it stores more than one value for the variable. Indeed it keeps a list of loan types.
There are two solutions to loose the minimum of information:
- either you choose to unfold the variable in several columns;
- or you choose to create several records with all the variables constant.
You have to choose between those options :-)
"""

df = dummify(df, ["Occupation"])
print(df.head(5))

df.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_encoded.csv", index=False)
