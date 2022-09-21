#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
from mmo import Domain, MultiModalMinimizer, compare_solutions
from modules_cec15 import fct
from hyperqueue import Job, LocalCluster
from scipy.spatial import distance_matrix
import sys

####################################################################################################
# config
####################################################################################################
FCT_NUM = int(sys.argv[1])
DIM = int(sys.argv[2])
PLOT = False

####################################################################################################
# functions
####################################################################################################
def extract_top_solutions(xy, dist):
    x = xy[:, :-1]
    y = xy[:, -1]
    idx = np.argsort(y)
    x = x[idx]
    y = y[idx]
    xy = xy[idx]

    idx = [0]
    for i in range(x.shape[0]):
        close = False
        for j in idx:
            if la.norm(x[i] - x[j]) < dist:
                close = True
                break
        if not close:
            idx += [i]
        if len(idx) == 5:
            break

    return(xy[idx])

####################################################################################################
# objective function and domain
####################################################################################################
f = fct(dim = DIM, fct_num = FCT_NUM)
dom = Domain(boundary = [f.ll, f.ur])

print("## configuration")
print(f'FCT_NUM = {FCT_NUM}')
print(f'DIM = {DIM}')
print(f'N_SOL = {f.n_sol}')
BUDGET = int(2000 * DIM * np.sqrt(f.n_sol)) 
print(f'BUDGET = {BUDGET}')
print(f'MINIMUM = {np.min(f.optima_xy[:, -1])}')
print(f'ECEX TIME: {f.exec_time:.3e} sec')
print(f'MIN DIST OPTIMA: {f.optima_mindist:.3f}')
print()

####################################################################################################
# run
####################################################################################################
mmm = MultiModalMinimizer(f = f, domain = dom, budget = BUDGET, verbose = 0)
for n_iter, iteration in enumerate(mmm):
    print(iteration)

xy = extract_top_solutions(mmm.pxy, DIM)
x = xy[:, :-1]
y = xy[:, -1]

d = distance_matrix(x, x)
print(d)
print()
print(y)








