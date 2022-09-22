#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
from mmo import Domain, MultiModalMinimizer, compare_solutions
from modules_cec13.cec2013 import how_many_goptima, CEC2013
from hyperqueue import Job, LocalCluster
from time import time
import sys

####################################################################################################
# config
####################################################################################################
PROBLEM = int(sys.argv[1])

####################################################################################################
# objective function and domain
####################################################################################################
cec = CEC2013(PROBLEM)
DIM = cec.get_dimension()
N_SOL = cec.get_no_goptima()
BUDGET = cec.get_maxfes()
OPTIMUM = cec.get_fitness_goptima()

# optima
if PROBLEM > 10:
    solutions = np.loadtxt('modules_cec13/data/optima.dat')
    solutions = solutions[:N_SOL, :DIM]
else:
    solutions = np.loadtxt(f'modules_cec13/data/optima_problem_{PROBLEM}')
    if len(solutions.shape) == 1:
        solutions = solutions.reshape(1, -1)
    solutions = solutions[:, :DIM]

# function and domain
LL = []
UR = []
for d in range(DIM):
    LL += [cec.get_lbound(d)]
    UR += [cec.get_ubound(d)]
f = lambda x: -cec.evaluate(x)
dom = Domain(boundary = [LL, UR])

print("## configuration")
print(f'PROBLEM = {PROBLEM}')
print(f'DIM = {DIM}')
print(f'N_SOL = {N_SOL}')
print(f'BUDGET = {BUDGET}')
print(f'OPTIMUM = {OPTIMUM}')
print()

####################################################################################################
# run
####################################################################################################
mmm = MultiModalMinimizer(f = f, domain = dom, budget = BUDGET, convergence = solutions)
for n_iter, iteration in enumerate(mmm):
    print(iteration)

####################################################################################################
# peak rate
####################################################################################################
CEC_f = CEC2013(PROBLEM)
n_optima = CEC_f.get_no_goptima()
count = np.zeros((5))
count[0], seeds = how_many_goptima(iteration.x, CEC_f, 1e-1)
count[1], seeds = how_many_goptima(iteration.x, CEC_f, 1e-2)
count[2], seeds = how_many_goptima(iteration.x, CEC_f, 1e-3)
count[3], seeds = how_many_goptima(iteration.x, CEC_f, 1e-4)
count[4], seeds = how_many_goptima(iteration.x, CEC_f, 1e-5)
peake_rate = np.mean(count) / n_optima

print("#####################")
print("## N solutions")
print("#####################")
print(PROBLEM, peake_rate, iteration.n_fev, mmm.convergence.contraction())
print()







