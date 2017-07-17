from ccpr.control_chart import control_chart as ast
import matplotlib.pyplot as plt
print("Usage: ControlChart(window, param, imbalance_ratio, 'abtype')")

print("abtypes: uptrend, downtrend, upshift, downshift, cyclic, systematic, stratified")
abtype = input("Enter abnormal pattern type: ")
window = int(input("Enter window length: "))
imbalance = int(input("Enter imbalance ratio: "))
param = float(input("Enter parameter of abnormal pattern: "))

ts = ast.ControlChart(window, param, imbalance, abtype)
ts.to_csv()

plt.plot(range(ts.get_window()), ts.get_norm(), 'r', label="normal")
plt.plot(range(ts.get_window()), ts.get_abnorm(), 'b', label="abnormal")
plt.axhline(y=ts.get_ucl(), color='green', linestyle='-', label="UCL")
plt.axhline(y=ts.get_lcl(), color='yellow', linestyle='-', label="LCL")
plt.ylabel("Observation")
plt.xlabel=("Time")
plt.legend()
plt.title(abtype)
plt.show()
