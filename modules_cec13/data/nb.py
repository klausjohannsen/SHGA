#!/usr/bin/env python

# libs
import numpy as np
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors

# own
import modules.parallel as parallel
import modules.basic as basic
from modules import Domain, CMA2, Function

####################################################################################################
# configuration
####################################################################################################
DIM = 5

if DIM == 2:
    NP = 6
elif DIM == 3:
    NP = 10
elif DIM == 5:
    NP = 20
elif DIM == 10:
    NP = 66
elif DIM == 20:
    NP = 230
else:
    assert(0)

N_GENE = NP//2 * (DIM - 1)

print("== NB ==")
print("DIM: %d" % DIM)
print("NP: %d" % NP)
print("N_GENE: %d" % N_GENE)

LL = [-1] * N_GENE
UR = [1] * N_GENE
dom = Domain(boundary=[LL, UR])

def g2x(g):
    x = g.reshape(-1, DIM - 1)
    x_norm = la.norm(x, axis=1).reshape(-1,1)
    xn = np.zeros((x.shape[0],1))
    for k in range(x.shape[0]):
        norm = la.norm(x[k])
        if norm < 1:
            xn[k,0] = np.sqrt(1 - norm*norm)
        else:
            x[k] = x[k]/norm
            xn[k,0] = 0
    x = np.hstack((x, xn))
    x = np.vstack((x, -x))
    return(x)


class f:
    def __init__(self):
        self.dim = N_GENE

    def __call__(self, g):
        x = g2x(g)
        xx = np.dot(x, np.transpose(x))
        np.fill_diagonal(xx, 0)
        r = np.array([np.max(xx)]) 

        return(r)

fct = f()

cma = CMA2(f=fct, domain=dom, tol=1e-2, verbose=1)

x0 = np.zeros(N_GENE)
g,y = cma.solve(x0, 1)

x = g2x(g)
xx = np.dot(x, np.transpose(x))

print(x)
print()
print(xx)
print()
print(y)

np.savetxt("data/nb_%.2d" % DIM, x)

