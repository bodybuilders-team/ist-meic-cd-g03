
"""
Regarding data partition, remember that time series are temporal data, and so test data shall
always be posterior to any train data. Remember, that Persistence model predicts the
following value based on the last one known, so we can consider two scenarios: the best –
corresponding to the one-step horizon, and the rough one – when we use the last value of
the training set to predict all the future values. In this manner, this model provides us two
baselines for comparing all the other results.
"""

# TODO: To be implemented

