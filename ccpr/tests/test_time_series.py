from ccpr.time_series import time_series as ast
import matplotlib.pyplot as plt
print("Usage: TimeSeries(window, param, imbalance_ratio, 'abtype')")

abtype = input("Enter abnormal pattern type: ")
window = int(input("Enter window length: "))
imbalance = int(input("Enter imbalance ratio: "))
param = float(input("Enter parameter of abnormal pattern: "))

upt = ast.TimeSeries(window, param, imbalance, abtype)
#upt.to_csv()

plt.plot(range(upt.get_window()), upt.get_norm()[0], 'r')
plt.plot(range(upt.get_window()), upt.get_abnorm()[0], 'b')
plt.show()
