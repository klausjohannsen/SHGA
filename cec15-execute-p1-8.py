#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
from mmo import Domain, MultiModalMinimizer, compare_solutions
from modules_cec15 import fct
from hyperqueue import Job, LocalCluster
import sys

####################################################################################################
# config
####################################################################################################
FCT_NUM = int(sys.argv[1])
DIM = int(sys.argv[2])
PLOT = False

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
print()

####################################################################################################
# run
####################################################################################################
for n_iter, iteration in enumerate(MultiModalMinimizer(f = f, domain = dom, budget = BUDGET)):
    print(iteration)
    if PLOT:
        iteration.plot()

try:
    iteration
    print(f'SOLUTIONS FOUND: {iteration.n_sol}')
except NameError:
    print(f'SOLUTIONS FOUND: 0')










