import numpy as np
from scipy.sparse import csr_array
from scipy.linalg import svd

def compute_projector(table, n=None, ns=None):
    if n is None:
        n = len(table)
    if ns is None:
        ns = np.max(table)
    hn_sizes = np.bincount(table)
    data = (np.array([hn_sizes[table[i]] for i in range(n)]))**(-1/2)
    C = csr_array((data, (np.arange(n), table)), shape=(n,ns))
    Pi = C @ C.T
    return Pi

def spectral_error(S, Pi, Uk, Lambdask_halfinv, method="RIP"):
    if method=="pseudoRIP":
        error = np.abs(np.linalg.norm(S @ Pi @ Uk @ Lambdask_halfinv, ord=2)-1)
    elif method=="RIP":
        sv = svd(S @ Pi @ Uk @ Lambdask_halfinv, compute_uv=False)
        upper_eps = np.max(sv)-1
        lower_eps = 1-np.min(sv)
        error = max(lower_eps, upper_eps)
    elif method=="RSA":
        error = np.linalg.norm(S @ (np.eye(Pi.shape[0]) - Pi) @ Uk @ Lambdask_halfinv, ord=2)
    else:
        raise ValueError("method argument should be 'RIP', 'pseudoRIP' or 'RSA', got {}".format(method))
    return error



    