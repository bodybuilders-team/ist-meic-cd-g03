from utils.dslabs_functions import run_linear_regression_study

covid_file_tag: str = "covid"
index_col: str = "date"
target: str = "deaths"

run_aggregation_study = True
run_smoothing_study = True

"""
------------------
Aggregation
------------------

% Approach 1: Weekly Aggregation
% Approach 2: Monthly Aggregation
% Approach 3: Quarterly Aggregation
"""

if run_aggregation_study:
    run_linear_regression_study(
        filename="../../data/covid/processed_data/forecast_covid_deaths_weekly_aggregated.csv",
        file_tag=covid_file_tag,
        index_col=index_col,
        target=target,
        title="Weekly Aggregation"
    )
    run_linear_regression_study(
        filename="../../data/covid/processed_data/forecast_covid_deaths_monthly_aggregated.csv",
        file_tag=covid_file_tag,
        index_col=index_col,
        target=target,
        title="Monthly Aggregation"
    )
    run_linear_regression_study(
        filename="../../data/covid/processed_data/forecast_covid_deaths_quarterly_aggregated.csv",
        file_tag=covid_file_tag,
        index_col=index_col,
        target=target,
        title="Quarterly Aggregation"
    )

"""
------------------
Smoothing
------------------

% Smoothing Sizes: 25, 50, 75, 100
"""

if run_smoothing_study:
    sizes: list[int] = [25, 50, 75, 100]
    for i in range(len(sizes)):
        run_linear_regression_study(
            filename=f"../../data/covid/processed_data/forecast_covid_deaths_smoothed_size_{sizes[i]}.csv",
            file_tag=covid_file_tag,
            index_col=index_col,
            target=target,
            title=f"Smoothing Size {sizes[i]}"
        )
