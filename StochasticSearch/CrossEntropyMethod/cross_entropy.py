import math
import matplotlib.pyplot as plt
from numpy.random import *
import  numpy.random as random
from scipy.spatial.distance import pdist
import numpy as np

class cross_entropy:
    def __init__(self, costs, c, rho, d, alpha, seed):
        """Initializes the CrossEntropyTSP class."""
        self.costs = costs
        self.n = len(costs)
        self.N = c * self.n**2
        self.rho = rho
        self.d = d
        self.alpha = alpha
        self.random_state = np.random.RandomState(seed)