from ccpr.control_chart import gen_data
import matplotlib.pyplot as plt
print("Usage: ControlChart(window, mean, param, imbalance_ratio, 'abtype', data_points)")

print("abtypes: uptrend, downtrend, upshift, downshift, cyclic, systematic, stratified, mix")
abtype = input("Enter abnormal pattern type: ")
window = int(input("Enter window length: "))
imbalance = int(input("Enter imbalance ratio: "))
mu = float(input("Enter mean of generated data: "))
param = float(input("Enter parameter of abnormal pattern: "))
dp = int(input("Enter number of data points: "))
#out = input("Enter name of output file: ")

ts = gen_data.ControlChart(window, mu, param, imbalance, abtype, dp)
ts.to_csv()

# plt.plot(range(ts.get_window()), ts.get_norm(), 'r', label="normal")
# plt.plot(range(ts.get_window()), ts.get_abnorm(), 'b', label="abnormal")
# plt.axhline(y=ts.get_ucl(), color='green', linestyle='-', label="UCL")
# plt.axhline(y=ts.get_lcl(), color='yellow', linestyle='-', label="LCL")
# plt.ylabel("Observation")
# plt.xlabel=("Time")
# plt.legend()
# plt.title(abtype)
# plt.show()
