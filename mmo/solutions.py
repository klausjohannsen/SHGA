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
import pandas as pd
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors

from mmo.integrate import xy_add, xy_reduce_xequal_ymin

###############################################################################
# function class
###############################################################################
class CircleRegions:
    def __init__(self, domain=None, solutions=None):
        self.domain = domain
        self.dim = domain.dim
        self.points = None
        self.radius = None
        self.solutions = solutions

    def set(self, population, percentile=-1):
        self.radius = None
        self.points = None
        if population is None:
            return
        self.population = population[:,:self.dim]
        self.points = self.solutions.sol[:,:self.dim]
        if self.points is not None and self.points.shape[0]>0:
            if self.population is not None and self.population.shape[0]>0:
                self.radius = np.zeros(self.points.shape[0])
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.points)
                distances, indices = nbrs.kneighbors(self.population)
                indices = indices.reshape(-1)
                distances = distances.reshape(-1)
                for k in range(self.points.shape[0]):
                    idx = indices==k
                    if sum(idx):
                        self.radius[k] = np.percentile(distances[idx], percentile)
                    else:
                        self.radius[k] = 0

    def get(self):
        return(self.points, self.radius)

class Solutions():
    def __init__(self, domain = None, xtol = None, true_solutions = None, plot = False, verbose = False):
        self.domain = domain
        self.dim = domain.dim
        self.verbose = verbose
        if xtol is None:
            self.xtol = 1e-6 * domain.diameter
        else:
            self.xtol = xtol
        self.n_sol_previous = 0
        self.n_sol = 0
        self.n_duplicates = 0
        self.n_duplicates_previous = 0
        self.sol = np.zeros((0, self.dim+1))
        self.true_solutions = true_solutions
        self._plot = plot
        self._plot_seeds = False
        self.seeds = None
        self.radius = None
        self.regions = None
        self.n_true_solutions = 0
        if true_solutions is not None:
            self.true_solutions = true_solutions
            self.n_true_solutions = true_solutions[1]

        if self.verbose:
            print(self)

    def set_radius(self, population=None):
        percentile = 50
        self.regions = CircleRegions(domain=self.domain, solutions=self)
        self.regions.set(population, percentile=percentile)

    def store_previous(self):
        self.n_sol_previous = self.n_sol
        self.n_duplicates_previous = self.n_duplicates

    def add(self, x, y):
        if x is None: return
        suggested_xy = np.hstack((x.reshape(-1, self.dim), y.reshape(-1, 1)))
        n_suggested = suggested_xy.shape[0]
        n_old = self.sol.shape[0]
        self.sol = xy_add(self.sol, suggested_xy)
        self.sol = xy_reduce_xequal_ymin(self.sol, tol = self.xtol)
        n_new = self.sol.shape[0]
        self.n_sol = n_new
        self.n_duplicates += n_suggested + n_old - n_new

    def solution_in_range(self, x, radius):
        for k in range(self.sol.shape[0]):
            if la.norm(x - self.sol[k,:self.dim]) < radius:
                return(True)
        return(False)

    def plot(self, population=None, out='screen', f=None):
        if self._plot == 0: return
        if self.dim == 1:
            self.plot_1d(out=out, population=population, f=f)
        elif self.dim == 2:
            self.plot_2d(out=out, population=population)
        else:
            self.plot_parallel_coordinates(out=out, population=population)

    def plot_1d(self, population=None, out='screen', f=None):
        fig, axs = plt.subplots(1, 1, figsize=(16,9))
        plt.xlim((self.domain.ll[0],self.domain.ur[0]))
        plt.title("Solutions")

        # function
        if f is not None:
            x = np.linspace(self.domain.ll[0], self.domain.ur[0], 1000)
            #f.record(False)
            y = np.zeros(1000)
            for k in range(1000):
                y[k] = f(x[k])
            #f.record(True)
            axs.plot(x, y, c='green')

        # population
        if population is not None and population.shape[0]>0:
            x = population[:,0]
            y = population[:,1]
            if x is not None:
                axs.scatter(x, y, s=30, c='lightgrey')

        # plot solutions
        if self.true_solutions is not None and self.true_solutions.shape[0]>0:
            x = self.true_solutions
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f(x[k])
                #f.record(True)
                axs.scatter(x[:,0], y, s=50, c='orange')

        # seeds
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0]>0:
            x = self.seeds[:,:self.dim]
            if x is not None:
                #f.record(False)
                y = f(x.reshape(1,-1))
                #f.record(True)
                axs.scatter(x[:,0], y, s=50, c='blue')

        # solutions
        if self.sol is not None and self.sol.shape[0]>0:
            x = self.sol[:,:self.dim]
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f(x[k])
                #f.record(True)
                axs.scatter(x[:,0], y, s=20, c='red')

        # global solutions
        if self.sol.shape[0]>0:
            x = self.sol[:, :self.dim]
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f(x[k])
                #f.record(True)
                axs.scatter(x[:,0], y, s=20, c='green')

        if out=='screen': plt.show()
        else:
            plt.savefig(out)
            plt.close()

    def plot_2d(self, population=None, out='screen'):
        fig, axs = plt.subplots(1, 1, figsize=(9,9))
        plt.xlim((self.domain.ll[0],self.domain.ur[0]))
        plt.ylim((self.domain.ll[1],self.domain.ur[1]))
        plt.title("Solutions")

        # population
        if population is not None and population.shape[0]>0:
            x = population[:,:self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=10, c='lightgrey')

        # plot solutions
        if self.true_solutions is not None and self.true_solutions.shape[0]>0:
            x = self.true_solutions
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=50, c='orange')

        # seeds
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0]>0:
            x = self.seeds[:,:self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=50, c='blue')

        # solutions
        if self.sol is not None and self.sol.shape[0]>0:
            x = self.sol[:,:self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=20, c='red')

        # global solutions
        if self.sol.shape[0]>0:
            x = self.sol[:, :self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=20, c='green')

        if out=='screen': plt.show()
        else:
            plt.savefig(out)
            plt.close()

    def plot_parallel_coordinates(self, out='screen', population=None):

        # init
        colors = []
        linewidth= []
        x = np.zeros((0,self.dim))
        plt.figure(figsize=(12,9))
        columns = ['x%d' % k for k in range(x.shape[1])]

        # true solutions
        if self.true_solutions is not None and self.true_solutions.shape[0]>0:
            x = np.vstack((x, self.true_solutions[:,:self.dim]))
            colors += ['orange']*self.true_solutions.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=8)
            ax.legend().remove()

        # population
        if population is not None and population.shape[0]>0:
            x = population[:,:self.dim]
            colors = ['lightgrey']*population.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=1)
            ax.legend().remove()

        # population in radius
        if self.radius is not None:
            x = population[:,:self.dim]
            y = self.sol[:,:self.dim]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
            distances, indices = nbrs.kneighbors(x)
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)
            radius = self.regions.radius[indices]
            idx = distances<radius

            x = population[idx,:self.dim]
            colors = ['grey']*population.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=1)
            ax.legend().remove()

        # seeds
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0]>0:
            x = self.seeds[:,:self.dim]
            colors = ['blue']*self.seeds.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=1)
            ax.legend().remove()

        # solutions
        if self.sol.shape[0]>0:
            x = self.sol[:,:self.dim]
            colors = ['red']*self.sol.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=2)
            ax.legend().remove()

        if out=='screen': plt.show()
        else:
            plt.savefig(out)
            plt.close()

    def __str__(self):
        s = 'solutions\n'
        if self.true_solutions is not None:
            s += f'  n_true_solutions: {self.n_true_solutions}\n'
        s += f'  x_tol: {self.xtol}\n'
        s += f'  n_sol: {self.n_sol} ({self.n_sol_previous})\n'
        s += f'  n_duplicates: {self.n_duplicates} ({self.n_duplicates_previous})\n'

        return(s)

    def min_dist_to_solution(self, population):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.solutions.sol)
        distances, indices = nbrs.kneighbors(population[:, :self.dim])
        distances = np.min(distances, axis=1).reshape(-1)
        return(distances)

    def population_wrt_regions(self, population):
        self.set_radius(population=population)
        if self.regions is None or self.regions.points is None or self.regions.points.shape[0] == 0:
            return(None, None)

        nbrs = NearestNeighbors(n_neighbors=self.regions.points.shape[0], algorithm='ball_tree').fit(self.regions.points)
        distances, indices = nbrs.kneighbors(population[:,:self.dim])
        idx = np.any(distances < self.regions.radius[indices], axis=1)
        idx_in = np.where(idx==True)[0].tolist()
        idx_out = np.where(idx==False)[0].tolist()
        return(idx_in, idx_out)











