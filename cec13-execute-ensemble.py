# libs
import numpy as np
from modules import hq_run_n

# fct
def geo_mean(x):
    r = np.exp(np.mean(np.log(x)))
    return(r)

# run
for PROBLEM in range(8, 21):
    r = hq_run_n(cmd = f'python3 cec13-execute-p1-20.py {PROBLEM}', n = 50, pp = 'last_4_float')
    print(f'n runs = {len(r)}')
    r = np.array(r)
    print(f'PROBLEM: {np.mean(r[:, 0])}')
    print(f'AVG FCT EVAL: {np.mean(r[:, 2])}')
    print(f'AVG CONTRACTION: {geo_mean(r[:, 3])}')
    print(f'PEAK RATE: {np.mean(r[:, 1])}')
    print()

    with open("results/results-cec13.txt", "a") as f:
        f.write(f'PROBLEM: {np.mean(r[:, 0])}, AVG FCT EVAL: {np.mean(r[:, 2])}, AVG CONTRACTION: {geo_mean(r[:, 3])}, PEAK RATE: {np.mean(r[:, 1])}\n')
    





