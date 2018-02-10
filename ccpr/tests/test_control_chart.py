import gen_data,learn
import matplotlib.pyplot as plt
print("Usage: ControlChart(window, mean, param, imbalance_ratio, 'abtype', data_points)")

print("abtypes: uptrend, downtrend, upshift, downshift, cyclic, systematic, stratified, mix")
abtype = input("Enter abnormal pattern type: ")
window = int(input("Enter window length: "))
imbalance = int(input("Enter imbalance ratio: "))
mu = float(input("Enter mean of generated data: "))
if abtype.strip() != "mix":
    param = float(input("Enter parameter of abnormal pattern: "))
else:
    param = 0
dp = int(input("Enter number of data points: "))
#out = input("Enter name of output file: ")

ts = gen_data.ControlChart(w=window, mu=mu, t=param, r=imbalance, abtype=abtype, data_points=dp, out="data")
#plot samp
plt.plot(range(ts.get_window()), ts.get_norm_samp(), 'r', label="normal")
plt.plot(range(ts.get_window()), ts.get_abnorm_samp(), 'b', label="abnormal")
plt.axhline(y=ts.get_ucl_samp(), color='green', linestyle='-', label="UCL")
plt.axhline(y=ts.get_lcl_samp(), color='yellow', linestyle='-', label="LCL")
plt.ylabel("Observation")
plt.xlabel=("Time")
plt.legend()
plt.title(abtype)
plt.show()
ts.to_csv()
data = learn.CC_Learn(filename="data.csv", norm=False)
data.input_normalize()
data.plot()
data.train()

