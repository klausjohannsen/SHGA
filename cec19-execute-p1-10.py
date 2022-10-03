#!/usr/bin/env python

# libraries
import numpy as np
import numpy.linalg as la
from mmo import Domain, MultiModalMinimizer
from cec2019comp100digit import cec2019comp100digit
import sys

####################################################################################################
# config
####################################################################################################
FCT = int(sys.argv[1])
BUDGET = 120000000

####################################################################################################
# solutions, objective function and domain
####################################################################################################
DIM = 10
LEN = 100
if FCT == 1: 
    DIM = 9
    LEN = 8192
if FCT == 2: 
    DIM = 16
    LEN = 16384
if FCT == 3: 
    DIM = 18
    LEN = 4

bench = cec2019comp100digit
bench.init(FCT, DIM) 
dom = Domain(boundary = [ [-LEN]*DIM, [LEN]*DIM ] )

####################################################################################################
# run
####################################################################################################
def digits(mmm):
    y = mmm.solutions.sol[:,-1]
    y_min = np.min(y)
    print(f'  min = {y_min:.10e}')
    d = y_min - 1
    digits = 0
    for k in range(10):
        if d < 10 ** (-k):
            digits = k + 1
        else:
            break
    print(f'  digits = {digits}')
    print()
    return(digits)

mmm = MultiModalMinimizer(f = bench.eval, domain = dom, budget = BUDGET, verbose = 1, fct_save_calls = False)
for n_iter, iteration in enumerate(mmm):
    print(iteration)
    if digits(mmm) >= 10:
        exit()

digits(mmm)







