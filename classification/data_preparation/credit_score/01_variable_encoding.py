import re

from pandas import DataFrame, read_csv
from sklearn.preprocessing import LabelEncoder

from utils.dslabs_functions import dummify

credit_score_filename: str = "../../data/credit_score/class_credit_score.csv"
credit_score_file_tag: str = "class_credit_score"
credit_score_data: DataFrame = read_csv(credit_score_filename, na_values="", index_col="ID")

# ------------------
# Ordinal Encoding: Binary and Categorical variables with order
# ------------------

# print ordinal variables and their values
# for col in credit_score_data.columns:
#     if credit_score_data[col].dtype == "object":
#         print(col, credit_score_data[col].unique())

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
credit_mix_type_values: dict[str, int] = {"Good": 0, "Standard": 1, "Bad": 2}
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
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

encoding: dict[str, dict[str, int]] = {
    "CreditMix": credit_mix_type_values,
    "Payment_of_Min_Amount": payment_of_min_amount_type_values,
    "Credit_Score": credit_score_type_values,
    "Payment_Behaviour": payment_behaviour_type_values,
    "Month": month_type_values,
}
df: DataFrame = credit_score_data.replace(encoding, inplace=False)

# SSN, Name and Customer_ID using label encoder
le = LabelEncoder()
df['SSN'] = le.fit_transform(df['SSN'])
df['Name'] = le.fit_transform(df['Name'])
df['Customer_ID'] = le.fit_transform(df['Customer_ID'])


# print(df.head(5))

# Credit_History_Age
def convert_year_month_to_decimal(age_str):
    if not isinstance(age_str, str):
        return None

    match = re.match(r'(\d+) Years and (\d+) Months', age_str)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return round(years + months / 12, 3)
    else:
        return None


df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_year_month_to_decimal)

# ------------------
# Dummification
# ------------------

"""
Column "Type_of_Loan" is "dummified", by creating a column for each loan type, and for each record, setting it to 
True if the loan type is present in the list of loan types, and False otherwise. Not "one-hot encoding" because 
there are records with more than one loan type, but a binary encoding nonetheless.

Resposta da stora:
The issue about the Type_of_Loan variable is that it stores more than one value for the variable. Indeed it keeps a list of loan types.
There are two solutions to loose the minimum of information:
- either you choose to unfold the variable in several columns; -> We chose this one.
- or you choose to create several records with all the variables constant.
You have to choose between those options :-)
"""

loan_types = ['Credit-Builder Loan', 'Home Equity Loan', 'Payday Loan', 'Debt Consolidation Loan', 'Personal Loan',
              'Auto Loan', 'Not Specified', 'Student Loan', 'Mortgage Loan']
for loan_type in loan_types:
    df[f"Type_of_Loan_{loan_type.replace(" ", "_")}"] = df['Type_of_Loan'].apply(
        lambda x: loan_type in x if isinstance(x, str) else False)

df.drop('Type_of_Loan', axis=1, inplace=True)  # Drop the original column

df = dummify(df, ["Occupation"])

print(df.head(5))

df.to_csv(f"../../data/credit_score/processed_data/{credit_score_file_tag}_encoded.csv", index=False)
