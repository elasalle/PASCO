import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import cgs, gmres, LinearOperator
from pygsp.graphs import Graph
from pygsp.filters.approximations import cheby_op

class FilterParameterError(Exception):
    pass

def create_combinatorial_lap(A):
    L = np.diag(np.sum(A, axis=1)) - A
    return L

def create_normalized_lap(A):
    # if not np.allclose(A, A.T):
    #     raise ValueError("create_normalized_lap: A should be symmetric")

    DEG_dem = diags(1 / np.sqrt(np.sum(A, axis=0)), 0)
    Q = eye(A.shape[0]) - DEG_dem @ A @ DEG_dem

    # # Make sure the Laplacian is exactly symmetrical
    # Qbis = np.triu(Q, 1)
    # Qbis = Qbis + Qbis.T
    # Ln = Qbis + diags(np.diag(Q), 0)
    Ln = Q

    # Replace any NaN values with 0
    Ln.data = np.nan_to_num(Ln.data)
    
    return Ln

def create_laplacian(A, lapversion):
    if lapversion == 'combinatorial':
        L = create_combinatorial_lap(A)
    elif lapversion == 'normalized':
        L = create_normalized_lap(A)
    else:
        raise ValueError("Laplacian type should be 'combinatorial' or 'normalized'")
    return L

def compute_jackson_cheby_coeff(filter_bounds, delta_lambda, m):
    r"""
    To compute the m+1 coefficients of the polynomial approximation of an ideal band-pass between a and b, between a range of values defined by lambda_min and lambda_max.

    Parameters
    ----------
    filter_bounds : list
        [a, b]
    delta_lambda : list
        [lambda_min, lambda_max]
    m : int

    Returns
    -------
    ch : ndarray
    jch : ndarray

    References
    ----------
    adapted from the function pygsp.filters.approximations.compute_jackson_cheby_coeff
    :cite:`tremblay2016compressive`

    """
    # Parameters check
    if delta_lambda[0] > filter_bounds[0] or delta_lambda[1] < filter_bounds[1]:
        raise FilterParameterError("Bounds of the filter are out of the lambda values")
    elif delta_lambda[0] > delta_lambda[1]:
        raise FilterParameterError("lambda_min is greater than lambda_max")

    # Scaling and translating to standard cheby interval
    a1 = (delta_lambda[1]-delta_lambda[0])/2
    a2 = (delta_lambda[1]+delta_lambda[0])/2

    # Scaling bounds of the band pass according to lrange
    filter_bounds[0] = (filter_bounds[0]-a2)/a1
    filter_bounds[1] = (filter_bounds[1]-a2)/a1

    # First compute cheby coeffs
    ch = np.zeros(m+1)
    ch[0] = (2/(np.pi))*(np.arccos(filter_bounds[0])-np.arccos(filter_bounds[1]))
    for i in range(1, m+1):
        ch[i] = (2/(np.pi * i)) * \
            (np.sin(i * np.arccos(filter_bounds[0])) - np.sin(i * np.arccos(filter_bounds[1])))

    # Then compute jackson coeffs
    jch = np.zeros(m+1)
    alpha = (np.pi/(m+2))
    for i in range(m+1):
        jch[i] = (1/np.sin(alpha)) * \
            ((1 - i/(m+2)) * np.sin(alpha) * np.cos(i * alpha) +
             (1/(m+2)) * np.cos(alpha) * np.sin(i * alpha))

    # Combine jackson and cheby coeffs
    jch = ch * jch

    return ch, jch

def estimate_lambda_k(pygspG, G, k, param=None):
    # Parameters
    if param is None:
        param = {}
    if 'nb_estimation' not in param:
        param['nb_estimation'] = 1
    if 'nb_features' not in param:
        param['nb_features'] = 2 * round(np.log(G['N']))
    if 'epsilon' not in param:
        param['epsilon'] = 1e-1
    if 'hint_lambda_max' not in param:
        param['hint_lambda_max'] = G['lmax']
    if 'hint_lambda_min' not in param:
        param['hint_lambda_min'] = 0
    if 'jackson' not in param:
        param['jackson'] = 1
    if 'order' not in param:
        param['order'] = 50
    
    # List of estimations for lambda_k
    norm_Uk = np.zeros((G['N'], param['nb_estimation']))
    lambda_k_est = np.zeros(param['nb_estimation'])
    
    # Perform nb_estimation on different sets of feature vectors
    for ind_est in range(param['nb_estimation']):
        # Random signals (fixed for one estimation)
        Sig = np.random.randn(G['N'], param['nb_features']) * (1 / np.sqrt(param['nb_features']))
        
        # Search by dichotomy
        counts = 0
        lambda_min = param['hint_lambda_min']
        lambda_max = param['hint_lambda_max']
        
        while (counts != k) and ((lambda_max - lambda_min) / lambda_max > param['epsilon']):
            # Middle of the interval
            lambda_mid = (lambda_min + lambda_max) / 2
            
            # Filter
            ch, jch = compute_jackson_cheby_coeff([0, lambda_mid], [0, G['lmax']], param['order'])
            
            # Filtering
            # if param['jackson'] == 2:
            #     X, Xjch = gsp_cheby_op2(G, ch.flatten(), jch.flatten(), Sig)
            #     countsch = np.round(np.sum(X * Sig))
            #     countsjch = np.round(np.sum(Xjch * Sig))
            #     counts = np.round((countsch + countsjch) / 2)
            # elif param['jackson'] == 1:
            if param['jackson'] == 1:
                X = cheby_op(pygspG, jch.flatten(), Sig)
                counts = np.round(np.sum(X ** 2))
            else:
                X = cheby_op(pygspG, ch.flatten(), Sig)
                counts = np.round(np.sum(X ** 2))
            
            # Update interval
            if counts > k:
                lambda_max = lambda_mid
            elif counts < k:
                lambda_min = lambda_mid
        
        # Store result
        lambda_k_est[ind_est] = (lambda_min + lambda_max) / 2
        norm_Uk[:, ind_est] = np.sum(X ** 2, axis=1)
    
    # Final estimation
    G['lk'] = np.mean(lambda_k_est)
    lambda_k = G['lk']
    cum_coh = np.mean(norm_Uk, axis=1)
    
    return G, lambda_k, cum_coh, X



def interpolate_on_complete_graph(c_obs, ind_obs, L, reg, N, solver):
    # Zero-fill c_obs
    b = np.zeros(N)
    b[ind_obs] = c_obs

    # Matrix to invert
    MtM = diags([1], [0], shape=(N, N))
    MtM = MtM.tocsr()
    MtM[ind_obs, ind_obs] = 1
    if callable(L):
        A = LinearOperator( (N,N) , lambda x :  MtM @ x + reg * L(x))
    else:
        MtM += reg * L
        A = MtM

    # Invert the system
    if solver == 'cgs':
        c_est, _ = cgs(A, b, rtol=1e-6, maxiter=100)
    elif solver == 'gmres':
        c_est, _ = gmres(A, b, rtol=1e-6, maxiter=100)
    else:
        raise ValueError("interpolate_on_complete_graph: solver must be either set to 'cgs' or to 'gmres'")

    return c_est
