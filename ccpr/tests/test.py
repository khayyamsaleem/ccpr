from ccpr import time_series

print("Usage: TimeSeries(window, param, imbalance_ratio, 'abtype')")

upt = TimeSeries(10, 0.105, 45, "uptrend")
upt.to_csv()
