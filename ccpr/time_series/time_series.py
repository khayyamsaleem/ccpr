import sys
from numpy.random import normal as randn
from numpy import pi
from numpy import cos
import pprint
import csv

class TimeSeries:
    def __init__(self, w, t, r, abtype):
        """Constructor for time series object"""
        self.__window = w
        self.__param = t
        self.__abtype = abtype.lower()
        self.__norm_size = (50+r)*10
        self.__abnorm_size = (50-r)*10

        self.__norm = [randn(0, 1, w).tolist() for _ in range(self.__norm_size)]
        g = randn(0, 1, w).tolist()

        abtypes = {
                "uptrend": lambda s: s+self.__param*g.index(s),
                "downtrend": lambda s: s-self.__param*g.index(s),
                "upshift": lambda s: s+self.__param,
                "downshift": lambda s: s-self.__param,
                "systematic": lambda s: s+self.__param*(-1)**g.index*(s),
                "cyclic": lambda s: s+self.__param*cos(3*pi*g.index(s)*0.125),
                "stratified": lambda s: s+self.__param*s
        }

        switch = lambda p: abtypes.get(p, -1)

        if(switch(self.__abtype) == -1):
            print("invalid abnormal pattern type")
            print("Valid types: " + str(list(abtypes.keys()))[1:-1])
            sys.exit(1)
        self.__abnorm = [list(map(switch(self.__abtype), g)) for _ in range(self.__abnorm_size)]

    def get_norm(self):
        return self.__norm

    def get_abnorm(self):
        return self.__abnorm

    def get_window(self):
        return self.__window

    def to_csv(self):
        def csv_writer(data, path):
            with open(path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for line in data:
                    writer.writerow(line)
        csv_writer(self.__norm, self.__abtype+"-normal.csv")
        csv_writer(self.__abnorm, self.__abtype+"-abnormal.csv")



if __name__ == "__main__":
    upt = TimeSeries(10, 0.105, 45, "uptrend")
    # pprint.PrettyPrinter(indent=4).pprint(upt.get_norm())
    cyc = TimeSeries(10, 0.5, 45, "cyclic")
    # pprint.PrettyPrinter(indent=4).pprint(cyc.get_abnorm())
    upt.to_csv()
    cyc.to_csv()

