#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
from mmo import Domain, MultiModalMinimizer, compare_solutions
from modules_cec13.cec2013 import how_many_goptima, CEC2013
from hyperqueue import Job, LocalCluster

####################################################################################################
# config
####################################################################################################
BUDGET = 100000
N_SOL = 25
DIM = 2
PLOT = True

####################################################################################################
# solutions, objective function and domain
####################################################################################################
solutions = np.random.rand(N_SOL, DIM)

def f(x):
    r = la.norm(solutions - x.reshape(1, -1), axis = 1)
    return(np.min(r))

dom = Domain(boundary = [ [0]*DIM, [1]*DIM ] )

####################################################################################################
# run
####################################################################################################
solutions_xy = np.zeros((solutions.shape[0], solutions.shape[1] + 1))
for k in range(solutions.shape[0]):
    solutions_xy[k, :DIM] = solutions[k]
    solutions_xy[k, DIM] = f(solutions[k])

mmm = MultiModalMinimizer(f = f, domain = dom, budget = BUDGET, max_sol = N_SOL, true_xy = solutions_xy)
for n_iter, iteration in enumerate(mmm):
    print(iteration)
    if PLOT:
        iteration.plot()










