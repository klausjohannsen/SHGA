#!/usr/bin/env python

# std libs
import warnings
import importlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon

import numpy as np
import numpy.linalg as la
import mmo.cma_ext as cma_ext
from mmo.cma_ext.fitness_functions import FitnessFunctions

###############################################################################
# function class
###############################################################################
class CMA():
    def __init__(self, f=None, domain=None, tol=1e-9, budget=None, verbose=0, centroid=None, popsize=None, diagonal=True):
        assert(f.dim==domain.dim)
        self.dim = f.dim
        self.f = f
        self.domain = domain
        self.tol = tol
        self.budget = budget
        self.n_fev = 0
        self.verbose = verbose
        self.centroid = centroid
        self.diagonal = diagonal
        if self.centroid is None:
            self.centroid = la.norm(self.domain.ll-self.domain.ur)/3

        # options for CMAEvolutionStrategy
        self.opts = cma_ext.CMAOptions()
        #self.opts.set('tolfun', self.tol)
        #self.opts.set('tolx', self.tol)
        self.opts.set('verb_disp', self.verbose)
        if popsize is not None:
            self.opts.set('popsize', popsize)

    def solve(self, seed, spread, corr_matrix):
        if self.diagonal == True:
            x = seed
            centroid = spread/3 if spread is not None else self.centroid
            assert(x.shape[0] == self.dim)
            es = cma_ext.CMAEvolutionStrategy(x.tolist(), centroid, self.opts)
            es.optimize(FitnessFunctions(f=self.f).ext)
            self.popsize = es.popsize

            # get solution path
            n_ensemble = int(es.result.evaluations/es.result.iterations)
            assert(n_ensemble==es.result.evaluations/es.result.iterations)
            xy = self.f.xy()
            xy = xy[-es.result.evaluations:,:]
            xy_split = np.split(xy, es.result.iterations)
            self.xy_path_ = np.zeros((len(xy_split), xy.shape[1]))
            for k, xy in enumerate(xy_split):
                self.xy_path_[k] = np.mean(xy, axis=0)

            # finish solve
            self.n_fev += es.result.evaluations
            return(es.result.xbest, es.result.fbest)

        else:
            assert(corr_matrix is not None)
            x0 = seed
            centroid = spread/3 if spread is not None else self.centroid
            assert(x0.shape[0] == self.dim)
            es = cma_ext.CMAEvolutionStrategy(x0.tolist(), centroid, self.opts)

            w, v = la.eig(corr_matrix)
            w = np.diag(w)
            Msqrt = v @ np.sqrt(w) @ v.T

            def f_transformed(x):
                x = x - x0.reshape(1,-1)
                x = x0 + Msqrt @ x.reshape(-1)
                x = x.reshape(1,-1)
                y = self.f(x)
                return(y)

            es.optimize(FitnessFunctions(f=f_transformed).ext)
            self.popsize = es.popsize

            # get solution path
            n_ensemble = int(es.result.evaluations/es.result.iterations)
            assert(n_ensemble==es.result.evaluations/es.result.iterations)
            xy = self.f.xy()
            xy = xy[-es.result.evaluations:,:]
            xy_split = np.split(xy, es.result.iterations)
            self.xy_path_ = np.zeros((len(xy_split), xy.shape[1]))
            for k, xy in enumerate(xy_split):
                self.xy_path_[k] = np.mean(xy, axis=0)

            # finish solve
            self.n_fev += es.result.evaluations
            xbest = es.result.xbest - x0
            xbest = x0 + Msqrt @ xbest
            return(xbest, es.result.fbest)

    def xy_path(self):
        return(self.xy_path_)

    def __str__(self):
        s = ''
        s += "## Nelder Mead ##\n"
        s += "dim: %d\n" % self.dim
        s += "n_fev: %d\n" % self.n_fev
        return(s)


