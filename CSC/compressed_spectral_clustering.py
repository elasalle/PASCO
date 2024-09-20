import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
from scipy.stats import rv_discrete
from CSC.CSC_utils import *
from sklearn.cluster import KMeans
from pygsp.graphs import Graph
from pygsp.filters.approximations import cheby_op
import time

def CSC(G, param_CSC, verbose=True):
    # Check parameter list
    if 'poly_order' not in param_CSC:
        param_CSC['poly_order'] = 50
    if 'regu' not in param_CSC:
        param_CSC['regu'] = 1e-3
    if 'sampling' not in param_CSC:
        param_CSC['sampling'] = 'uniform'
    if 'n_factor' not in param_CSC:
        param_CSC['n_factor'] = 2
    if 'd_factor' not in param_CSC:
        param_CSC['d_factor'] = 4
    if 'lap_type' not in param_CSC:
        param_CSC['lap_type'] = 'normalized'
    if 'solver' not in param_CSC:
        param_CSC['solver'] = 'gmres'

    # Compute required Laplacian matrix
    if param_CSC['lap_type'] == 'normalized':
        normBOOL = True
        G['lap_type'] = 'normalized'
        G['L'] = create_laplacian(G['W'], G['lap_type'])
    elif param_CSC['lap_type'] == 'combinatorial':
        normBOOL = False
        G['lap_type'] = 'combinatorial'
        G['L'] = create_laplacian(G['W'], G['lap_type'])
    else:
        raise ValueError('param_CSC.lap_type should be "normalized" or "combinatorial"')
    
    # encode the graph with pygsp
    pygspG = Graph(G['W'])
    pygspG.estimate_lmax()

    # Estimate lambda_k and weight VD
    timings = {}
    if normBOOL:
        G['lmax'] = 2
        timings['lmax'] = 0
    else:
        ti = time.time()
        opts = {'isreal': True, 'issym': True, 'maxit': 10000}
        G['lmax'] = eigs(G['L'], k=1, which='LM', **opts)[0][0].real
        timings['lmax'] = time.time() - ti

    if verbose:
        print('Estimating lambda_k...')
    ti = time.time()
    param = {'order': param_CSC['poly_order']}
    _, lk_est, cum_coh_k, _ = estimate_lambda_k(pygspG, G, G['k'], param)
    param['hint_lambda_max'] = lk_est*2
    _, lk_estp1, cum_coh_k_p1, _ = estimate_lambda_k(pygspG, G, G['k'], param)
    lk_est = (lk_est + lk_estp1) / 2
    timings['lk_est'] = time.time() - ti
    if param_CSC['sampling'] == 'VD':
        mean_num_coh = np.mean([cum_coh_k, cum_coh_k_p1], axis=0)
        weight_VD = mean_num_coh
        weight_VD = weight_VD / np.sum(weight_VD)
    else:
        weight_VD = None
    if verbose:
        print('\tDone.')

    # Filter d random vectors
    G['lk'] = lk_est

    if verbose:
        print('Filtering random signals...')
    n = round(param_CSC['n_factor'] * G['k'] * np.log(G['k']))
    d = round(param_CSC['d_factor'] * np.log(n))
    ti = time.time()
    R = (1 / np.sqrt(d)) * np.random.randn(G['N'], d)

    _, JCH = compute_jackson_cheby_coeff([0, G['lk']], [0, G['lmax']], param_CSC['poly_order'])
    X_lk_est = cheby_op(pygspG, JCH, R)

    if normBOOL:
        X_lk_est /= np.sqrt(np.sum(X_lk_est ** 2, axis=1, keepdims=True))
    timings['filtering'] = time.time() - ti
    if verbose:
        print('\tDone.')

    # Downsample n nodes
    if param_CSC['sampling'] == 'uniform':
        weight = np.ones(G['N']) / G['N']  # Uniform density
    elif param_CSC['sampling'] == 'VD':
        weight = weight_VD  # Variable density
    else:
        raise ValueError('param_CSC.sampling must be either set to "uniform" or to "VD"')

    ind_obs = rv_discrete(values=(range(G['N']), weight)).rvs(size=n)
    X_lk_est_DS = X_lk_est[ind_obs, :]

    # Do k-means in low dimension
    if verbose:
        print('Low-dimensional kmeans...')
    ti = time.time()
    kmeans = KMeans(G['k'], n_init=20).fit(X_lk_est_DS)
    IDX_LD = kmeans.labels_
    timings['k_means_low_dim'] = time.time() - ti
    if verbose:
        print('\tDone.')

    # Interpolate in high dimensions
    if verbose:
        print('Interpolation of cluster indicators...')
    ti = time.time()
    C_obs_LD = coo_matrix((np.ones(n), (range(n), IDX_LD)), shape=(n, G['k'])).toarray()
    _, JCH_HP = compute_jackson_cheby_coeff([G['lk'], G['lmax']], [0, G['lmax']], param_CSC['poly_order'])

    C_est = np.zeros((G['N'], C_obs_LD.shape[1]))

    for k in range(C_obs_LD.shape[1]):
        c_obs = C_obs_LD[:, k]
        C_est[:, k] = interpolate_on_complete_graph(c_obs, ind_obs, lambda x :  cheby_op(pygspG, JCH_HP, x), 
                                                    param_CSC['regu'], G['N'], param_CSC['solver'])
    timings['interpolation'] = time.time() - ti
    if verbose:
        print('\tDone.')

    timings['total'] = timings['interpolation'] + timings['lmax'] + timings['lk_est'] + timings['filtering'] + timings['k_means_low_dim']
    return C_est, lk_est, timings, IDX_LD, ind_obs, weight_VD
