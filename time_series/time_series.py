from random import randint
import numpy as np

class Time_Series:
    def __init__(self, w, t, a, abtype):
        """Constructor for time series object"""
        self.window = w
        self.param = t
        self.imbalance = a
        self.abtype = abtype
        self.x = range(w)
        self.y = [0]*w
        for i in self.x:

