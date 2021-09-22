import copy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys, os

sys.path.append('../')
from characteristic_functions import charac_function, charac_function_multiscale, charac_function_multiscale_hyper
from wave_utils.graph_tools import laplacian


TAUS = [1, 10, 25, 50]
ORDER = 30
PROC = 'approximate'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 2


def compute_cheb_coeff(scale, order):
    coeffs = [(-scale)**k * 1.0 / math.factorial(k) for k in range(order + 1)]
    return coeffs


def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1-2):
        basis.append(2* np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def heat_diffusion_ind(graph, taus=TAUS, order = ORDER, proc = PROC):

    # Compute Laplacian
    a = nx.adjacency_matrix(graph)
    n_nodes, _ = a.shape
    thres = np.vectorize(lambda x : x if x > 1e-4 * 1.0 / n_nodes else 0)
    lap = laplacian(a)
    n_filters = len(taus)
    if proc == 'exact':
        ### Compute the exact signature
        lamb, U = np.linalg.eigh(lap.todense())
        heat = {}
        for i in range(n_filters):
             heat[i] = U.dot(np.diagflat(np.exp(- taus[i] * lamb).flatten())).dot(U.T)
    else:
        heat = {i: sc.sparse.csc_matrix((n_nodes, n_nodes)) for i in range(n_filters) }
        monome = {0: sc.sparse.eye(n_nodes), 1: lap - sc.sparse.eye(n_nodes)}
        for k in range(2, order + 1):
             monome[k] = 2 * (lap - sc.sparse.eye(n_nodes)).dot(monome[k-1]) - monome[k - 2]
        for i in range(n_filters):
            coeffs = compute_cheb_coeff_basis(taus[i], order)
            heat[i] = sc.sum([ coeffs[k] * monome[k] for k in range(0, order + 1)])
            temp = thres(heat[i].A) # cleans up the small coefficients
            heat[i] = sc.sparse.csc_matrix(temp)
    return heat, taus


def graphwave_alg(graph, time_pnts, taus= [0.5], 
              verbose=False, approximate_lambda=True,
              order=ORDER, proc=PROC, nb_filters=NB_FILTERS,
              **kwargs):

    heat_print, _ = heat_diffusion_ind(graph, list(taus), order=order, proc = proc)
    return heat_print, taus

