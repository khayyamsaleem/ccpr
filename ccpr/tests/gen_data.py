from numpy.random import normal as randn
from numpy import pi, cos, mean, std
import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random

class ControlChart:
    def __init__(self, w, mu, t, r, abtype, data_points, out="data"):
        """Constructor for time series object"""
        self.__window = w
        self.__mean = mu
        self.__param = t
        self.__abtype = abtype.lower()
        self.__imbalance = r
        self.__norm_size = (50+r)*0.01
        self.__abnorm_size = (50-r)*0.01
        self.__data_points = data_points+1
        self.__out = out.strip()

        g = lambda : randn(0, self.__mean, self.__window).tolist()

        abnormal_types = ["uptrend", "downtrend", "upshift", "downshift", "systematic", "cyclic", "stratified"]
        default_params = {
            abnormal_types[0]: 0.8,
            abnormal_types[1]: 0.8,
            abnormal_types[2]: 3.2,
            abnormal_types[3]: 3.2,
            abnormal_types[4]: 1.8,
            abnormal_types[5]: 1.803,
            abnormal_types[6]: 1.2
        }
        def switch(p,randfn=g()):
            abtypes = {
                    abnormal_types[0] : lambda s: s+self.__param*randfn.index(s),
                    abnormal_types[1]: lambda s: s-self.__param*randfn.index(s),
                    abnormal_types[2]: lambda s: s+self.__param,
                    abnormal_types[3]: lambda s: s-self.__param,
                    abnormal_types[4]: lambda s: s+self.__param*(-1)**randfn.index(s),
                    abnormal_types[5]: lambda s: s + self.__param*cos(2*pi*randfn.index(s)*0.125),
                    abnormal_types[6]: lambda s: s+self.__param*s
            }
            if abtypes.get(p, -1) == -1:
                return False
            return list(map(abtypes.get(p, -1), randfn))

        self.__norm = [[0] + g() for _ in range(int(self.__norm_size*self.__data_points))]

        if self.__abtype == "mix":
            def mix_fn(f): 
                label = abnormal_types.index(f)
                self.__param = default_params[f]
                return [label] + switch(f, randfn=g())
            self.__abnorm = [mix_fn(random.choice(abnormal_types)) for _ in range(int(self.__abnorm_size*self.__data_points))]
        else:
            if not switch(self.__abtype):
                print("invalid abnormal pattern type")
                print("Valid types: " +str(abnormal_types+["mix"])[1:-1])
                sys.exit(1)

            self.__abnorm = [[1] + switch(self.__abtype, randfn=g()) for _ in range(int(self.__abnorm_size*self.__data_points))]

    def get_norm_samp(self):
        return self.__norm[0][1:]

    def get_abnorm_samp(self):
        return self.__abnorm[0][1:]

    def get_window(self):
        return self.__window

    def get_ucl_samp(self):
        return std(self.__norm[0][1:])*3

    def get_lcl_samp(self):
        return std(self.__norm[0][1:])*(-3)

    def get_std_dev_samp(self):
        return std(self.__norm[0][1:])

    def get_mean_samp(self):
        return mean(self.__norm[0][1:])

    def to_csv(self):
        def csv_writer(norm, abnorm, path):
            with open(path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for line in norm:
                    writer.writerow(line)
                for line in abnorm:
                    writer.writerow(line)
        csv_writer(self.__norm, self.__abnorm, self.__out+".csv")

