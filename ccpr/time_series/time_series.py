from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Time_Series:
    def __init__(self, w, t, a, abtype):
        """Constructor for time series object"""
        self.window = w
        self.param = t
        self.imbalance = a
        self.abtype = abtype
        
        mu, sigma = 0, 1
        s = np.random.normal(mu, sigma, 1000)

if __name__ == "__main__":
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 20, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 *np.pi)) * np.exp(-(bins-mu)**2 / (2*sigma**2)), linewidth=2, color = 'r')
    plt.show()
