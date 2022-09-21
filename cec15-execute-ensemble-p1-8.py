# libs
import numpy as np
from modules import hq_run_n
import sys

# run
TODO = [ [1, 5], [1, 10], [1, 20], [2, 2], [2, 5], [2, 8], [3, 2], [3, 3], [3, 4], [4, 5], [4, 10], [4, 20], [5, 2], [5, 3], [5, 4], [6, 4], [6, 6], [6, 8], [7, 6], [7, 10], [7, 16], [8, 2], [8, 3], [8, 4] ]

for FCT_NUM, DIM in TODO:
    print('+------------------+------------------------------+')
    print(f'+ cec15, run: FCT_NUM = {FCT_NUM},  DIM = {DIM}')
    r = hq_run_n(cmd = f'python3 cec15-execute-p1-8.py {FCT_NUM} {DIM}', n = 25, pp = 'last_int')
    print(f'n runs = {len(r)}')
    print(f'n_sols = {r}')
    print(f'min = {np.min(r)}, max = {np.max(r)}, mean = {np.mean(r)}, std = {np.std(r, ddof = 1)}')
    print()

    with open("results/results-cec15-p1-8.txt", "a") as f:
        f.write(f'FCT_NUM = {FCT_NUM},  DIM = {DIM}: n_sols = {r}, min = {np.min(r)}, max = {np.max(r)}, mean = {np.mean(r)}, std = {np.std(r, ddof = 1)}\n')





