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

        g = randn(0, self.__mean, self.__window).tolist()

        abtypes = {
                "uptrend": lambda s: s+self.__param*g.index(s),
                "downtrend": lambda s: s-self.__param*g.index(s),
                "upshift": lambda s: s+self.__param,
                "downshift": lambda s: s-self.__param,
                "systematic": lambda s: s+self.__param*(-1)**g.index(s),
                "cyclic": lambda s: s + self.__param*cos(2*pi*g.index(s)*0.125),
                "stratified": lambda s: s+self.__param*s
        }

        switch = lambda p: abtypes.get(p, -1)

        self.__norm = [[0] + randn(0, self.__mean, self.__window).tolist() for _ in range(int(self.__norm_size*self.__data_points))]

        if self.__abtype == "mix":
            self.__abnorm = [[1] + list(map(random.choice(list(abtypes.items())), g)) for _ in range(int(self.__abnorm_size*self.data_points))]
        else:
            if switch(self.__abtype) == -1:
                print("invalid abnormal pattern type")
                print("Valid types: " +str(list(abtypes.keys())+["mix"])[1:-1])
                sys.exit(1)

            self.__abnorm = [[1] + list(map(switch(self.__abtype), g)) for _ in range(int(self.__abnorm_size*self.__data_points))]

    # def get_norm(self):
    #     return self.__norm

    # def get_abnorm(self):
    #     return self.__abnorm

    # def get_window(self):
    #     return self.__window

    # def get_ucl(self):
    #     return std(self.__norm)*3

    # def get_lcl(self):
    #     return std(self.__norm)*(-3)

    # def get_std_dev(self):
    #     return std(self.__norm)

    # def get_mean(self):
    #     return mean(self.__norm)

    def to_csv(self):
        def csv_writer(norm, abnorm, path):
            with open(path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for line in norm:
                    writer.writerow(line)
                for line in abnorm:
                    writer.writerow(line)
        csv_writer(self.__norm, self.__abnorm, self.__out+".csv")

