#!/usr/bin/env python

# std libs
import warnings
import importlib
import copy

import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors
from time import time

###############################################################################
# functions
###############################################################################
def geo_mean(x):
    r = np.exp(np.mean(np.log(x)))
    return(r)

###############################################################################
# classes
###############################################################################
class Convergence:
    def __init__(self, solutions=None, n_min=None, n=None):
        assert(solutions is not None)
        assert(n_min is not None)
        self.solutions = solutions
        self.n_solutions = solutions.shape[0]
        self.n_min = n_min
        self.n = n
        self.dim = solutions.shape[1]
        self.dtypes = {'generation': 'int', 'solution_idx': 'int', 'distance': 'float'}
        self.df = pd.DataFrame(None, columns=['generation', 'solution_idx', 'distance']).astype(self.dtypes)
        self.generation = 0

    def add_old(self, population=None):
        if isinstance(population, list):
            population = np.array(population)
        population = population[:,:self.dim]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.solutions)
        distances, indices = nbrs.kneighbors(population)
        for k in range(distances.shape[0]):
            row = {'generation': self.generation, 'solution_idx': indices[k,0], 'distance': distances[k,0]}
            self.df = self.df.append(row, ignore_index=True)
        self.df = self.df.astype(self.dtypes)
        self.generation += 1

    def add(self, population=None):
        if isinstance(population, list):
            population = np.array(population)
        population = population[:,:self.dim]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.solutions)
        distances, indices = nbrs.kneighbors(population)
        arr = np.vstack((self.generation * np.ones(distances.shape[0]), indices[:,0], distances[:,0])).T
        self.df = pd.concat([self.df, pd.DataFrame(arr, columns=['generation', 'solution_idx', 'distance'])])
        #self.df = self.df.astype(self.dtypes)
        self.generation += 1

    def contraction(self):
        n_gen = int(self.df['generation'].max())
        x = np.zeros((0,2))
        for k in range(n_gen):
            df1 = self.df[self.df['generation'] == k][['solution_idx', 'distance']]
            df2 = self.df[self.df['generation'] == k + 1][['solution_idx', 'distance']]
            for kk in range(self.n_solutions):
                df1s = df1[df1['solution_idx'] == kk][['distance']].sort_values('distance')
                df2s = df2[df2['solution_idx'] == kk][['distance']].sort_values('distance')
                n = min(len(df1s), len(df2s))
                if n < self.n_min:
                    continue
                x1 = np.array(df1s.head(n))
                x2 = np.array(df2s.head(n))
                x_max = np.max(x1)
                x1 /= x_max
                x2 /= x_max
                x = np.vstack((x, np.hstack((x1, x2))))

        xx = np.zeros((x.shape[0], 2))
        n = 0
        for k in range(x.shape[0]):
            if x[k, 0] > 0 and x[k, 1] > 0:
                xx[n] = x[k]
                n += 1
        x = xx[:n]

        y = x[:,1] / x[:,0]
        r = geo_mean(y ** self.n)

        return(r)

    def __add__(self, other):
        assert(self.n_min == other.n_min)
        assert(self.n == other.n)
        assert(self.dim == other.dim)
        sum = copy.deepcopy(self)
        sum.df = pd.concat([self.df, other.df])

        return(sum)



















