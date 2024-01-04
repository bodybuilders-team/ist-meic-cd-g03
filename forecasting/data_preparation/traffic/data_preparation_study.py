import time

from utils.dslabs_functions import run_linear_regression_study

traffic_file_tag: str = "forecast_traffic"
index_col: str = "Timestamp"
target: str = "Total"

run_aggregation_study = True
run_smoothing_study = True
run_differentiation_study = True

"""
------------------
Aggregation
------------------

% Approach 1: Hourly Aggregation
% Approach 2: Daily Aggregation
% Approach 3: Weekly Aggregation
"""
if run_aggregation_study:
    run_linear_regression_study(
        filename="../../data/traffic/processed_data/forecast_traffic_hourly_aggregated.csv",
        file_tag=traffic_file_tag,
        index_col=index_col,
        target=target,
        title="Hourly Aggregation"
    )
    time.sleep(5)
    run_linear_regression_study(
        filename="../../data/traffic/processed_data/forecast_traffic_daily_aggregated.csv",
        file_tag=traffic_file_tag,
        index_col=index_col,
        target=target,
        title="Daily Aggregation"
    )
    time.sleep(5)
    run_linear_regression_study(
        filename="../../data/traffic/processed_data/forecast_traffic_weekly_aggregated.csv",
        file_tag=traffic_file_tag,
        index_col=index_col,
        target=target,
        title="Weekly Aggregation"
    )
    time.sleep(5)

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
            filename=f"../../data/traffic/processed_data/forecast_traffic_smoothed_size_{sizes[i]}.csv",
            file_tag=traffic_file_tag,
            index_col=index_col,
            target=target,
            title=f"Smoothing Size {sizes[i]}"
        )
        time.sleep(5)

"""
------------------
Differentiation
------------------

% Approach 1: First differentiation
% Approach 2: Second differentiation
"""
if run_differentiation_study:
    run_linear_regression_study(
        filename="../../data/traffic/processed_data/forecast_traffic_first_diff.csv",
        file_tag=traffic_file_tag,
        index_col=index_col,
        target=target,
        title="First Differentiation"
    )
    time.sleep(5)
    run_linear_regression_study(
        filename="../../data/traffic/processed_data/forecast_traffic_second_diff.csv",
        file_tag=traffic_file_tag,
        index_col=index_col,
        target=target,
        title="Second Differentiation"
    )
    time.sleep(5)
