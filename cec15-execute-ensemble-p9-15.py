# libs
import numpy as np
from modules import hq_run_n

# run
for FCT_NUM in range(9, 16):

    r = hq_run_n(cmd = f'cec15-execute-p9-15.py {FCT_NUM} 10', n = 25, pp = 'last_5_float')
    print()
    x10 = np.array(r)
    print(x10)

    r = hq_run_n(cmd = f'cec15-execute-p9-15.py {FCT_NUM} 20', n = 25, pp = 'last_5_float')
    print()
    x20 = np.array(r)
    print(x20)

    r = hq_run_n(cmd = f'cec15-execute-p9-15.py {FCT_NUM} 30', n = 25, pp = 'last_5_float')
    print()
    x30 = np.array(r)
    print(x30)

    print()
    print("all runs 25 * [10D, 20D, 30D] = 75")
    x = np.vstack((x10, x20, x30))
    print()
    print("median")
    xm = np.median(x, axis = 0).reshape(-1,1)
    print(xm)

    with open("results/results-cec15-p9-15.txt", "a") as f:
        f.write(f'FCT_NUM = {FCT_NUM}: median solutions = {xm.reshape(-1)}\n')







