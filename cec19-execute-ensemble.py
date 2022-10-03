# libs
import numpy as np
from modules import hq_run_n

# run
#for FCT in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
for FCT in [ 8]:
    print(f'FCT: {FCT}')
    r = hq_run_n(cmd = f'python3 cec19-execute-p1-10.py {FCT}', n = 50, pp = 'last_int')
    print(f'n runs = {len(r)}')
    r = np.array(r)
    r = np.flip(np.sort(r))
    r = r[:25]
    print(r[:15])
    print()
    r_mean = np.mean(r)
    print(f'r_mean = {r_mean}')

    with open("results/results-cec19.txt", "a") as f:
        f.write(f'FCT: {FCT}, r = {r}, r_mean = {r_mean}\n')
    





