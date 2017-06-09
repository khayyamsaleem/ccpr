from ccpr.time_series import time_series as ast
print("Usage: TimeSeries(window, param, imbalance_ratio, 'abtype')")

upt = ast.TimeSeries(10, 0.105, 45, "uptrend")
upt.to_csv()
