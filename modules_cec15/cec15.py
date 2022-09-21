import ctypes
from ctypes import c_void_p, c_double, c_int, cdll
import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
from os.path import exists
from time import time
from scipy.spatial import distance_matrix

# class
class fct:
    def __init__(self, dim = 2, fct_num = 1):
        self.dim = dim
        self.ll = [-100] * self.dim
        self.ur = [100] * self.dim
        self.fct_num = fct_num
        self.r = np.array([0.0])
        self.init = 1

        self.lib = cdll.LoadLibrary("./modules_cec15/cec15_nich_func.so")
        self.niche_fct = self.lib.cec15_nich_func

        self.func_num = fct_num
        self.nx = self.dim
        self.mx = 1

        optima_file = f'modules_cec15/optima/optima_positions_{self.fct_num}_{self.dim}D.txt'
        self.optima = np.loadtxt(optima_file)
        self.optima_xy = np.zeros((self.optima.shape[0], self.optima.shape[1] + 1))
        self.n_sol = self.optima.shape[0]

        self.ll = np.array(self.ll)
        self.ur = np.array(self.ur)

        # solutions
        for k in range(self.n_sol):
            self.optima_xy[k, :-1] = self.optima[k]
            self.optima_xy[k, -1] = self(self.optima[k])

        # get execution time
        t = 0
        for k in range(1000):
            x = self.ll + (self.ur - self.ll) * np.random.rand(self.dim)
            t0 = time()
            y = self(x)
            t += time() - t0
        self.exec_time = t / 1000

        # minimum distances of optima
        d = distance_matrix(self.optima, self.optima)
        np.fill_diagonal(d, np.inf)
        self.optima_mindist = np.min(d)

    def __call__(self, x):
        x = x.astype(float)
        self.niche_fct(c_void_p(x.ctypes.data), c_void_p(self.r.ctypes.data), c_int(self.nx), c_int(self.mx), c_int(self.func_num), c_int(self.init))
        self.init = 0
        return(self.r[0])


