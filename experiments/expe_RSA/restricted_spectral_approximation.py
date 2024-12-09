from pasco.data_generation import generate_or_import_SBM
from pasco.coarsening import Coarsening
from utils import compute_projector, spectral_error
from libraries.coarsening_utils import coarsen
from libraries import graph_lib
import numpy as np
import networkx as nx
from pygsp.graphs import Graph
from scipy.sparse import csr_array
from scipy.linalg import  eigh
from scipy.sparse.linalg import eigsh
from time import time, perf_counter
import pickle
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':

    # hyper-parameters
    graph_types = ["real", "SBM"] 
    # rs = np.linspace(0.1,0.9,2)
    rs = np.linspace(0.1,0.9,17)
    n_repetitions = 10  # number of timmes the coarsening is repeated
    method_sampling = 'uniform_node_sampling'
    parallel = False
    verbose = False
    save_fig = True
    coarsening_methods = ["pasco", 'variation_edges', 'heavy_edge']
    labels = {"pasco":"PASCO",
              'variation_edges':'variation_edges',
              'heavy_edge':'heavy_edge'}
    err_type = "RSA"

    res_dir = "results/"
    saving_file_name = res_dir + '/results' + '.pickle'

    errors = {}
    timings = {}
    for graph_type in graph_types:
        if graph_type=="SBM":
            # (n,k,d,alpha)
            # graphs = [(1000,10,10,1/10), (1000,10,10,1/20), (1000,100,10,1/100), (1000,100,10,1/200)]
            graphs = [(1000,10,2,1/10), (1000,100,2,1/100), (1000,100,2,1/200)]
        else:
            graphs  = ['yeast','minnesota', 'airfoil'] 
        
        # Data generation
        As, Ls = [], []
        if graph_type=="SBM":
            for n,k,d,alpha in graphs:
                n_k = n // k
                avg_d = d * np.log(n)
                pin = avg_d / ((1 + (k-1)*alpha )*n_k)
                pout = alpha * pin
                G = generate_or_import_SBM(n, k, pin, pout, data_folder="../data/graphs/SBMs/",  seed=2024)
                A = nx.adjacency_matrix(G, nodelist=range(n))
                As.append(A)
                L = nx.laplacian_matrix(G, nodelist=range(n)) # combinatorial laplacian
                Ls.append(L)
            del G
        else:
            k = 10 # size of the subspace to preserve in RSA
            for graph in graphs:
                G = graph_lib.real(10000, graph)
                As.append(csr_array(G.W))
                Ls.append(G.L)
        
        
        errors[graph_type] = {}
        timings[graph_type] = {}
        for A,L,graph in zip(As, Ls, graphs):
            print("Graph : {}".format(graph))

            n = A.shape[0]
            errors[graph_type][graph] = {}
            timings[graph_type][graph] = {}


            # compute the square root of the laplacian
            lambdas, U = eigh(L.toarray())
            lambdas[0] = 0
            S = U @ np.diag(np.sqrt(lambdas)) @ U.T
            # S = cholesky(L.todense())
            print("S computed")
            print("S.T @ S == L :  {}".format(np.linalg.norm(L - S.T @ S) < 1e-5 ))

            # compute the k first (without the first) eigen elements (used for the error computations)
            lambdas_k, Uk = eigsh(csr_array._asfptype(L), k=k, which='SM', tol=1e-6)
            Lambdask_halfinv = np.diag(lambdas_k**(-1/2))
            Lambdask_halfinv = Lambdask_halfinv[1:, 1:] # remove the first element
            Uk = Uk[:,1:] # remove the first element
            # Lambdask_halfinv[0] = 0 # enforce the first eigenvalue to be 0
            print("eigen elements computed")

            
            for cm in coarsening_methods:
                print("coarsening method: {}".format(cm))
                errors[graph_type][graph][cm] = []
                timings[graph_type][graph][cm] = []
                for r in rs:
                    ns = int(np.floor((1-r)*n))
                    print("  r = {:.0%}".format(r))
                    Pis = []
                    times = []
                    for _ in range(n_repetitions):
                        pygspG = Graph(A)
                        if cm=="pasco":
                            ti = perf_counter()
                            coarsening = Coarsening(ns, 1)
                            Aprimes, tables = coarsening.fit_transform_multiple(A, n_tables=1)
                            tf = perf_counter()
                            Pi = compute_projector(tables[0], n, ns)
                        else:
                            ti = perf_counter()
                            C, _, _, _ = coarsen(pygspG, K=k, r=r, max_levels=20, method=cm, algorithm='greedy')
                            tf = perf_counter()
                            Pi = C.T @ C
                        Pis.append(Pi)
                        times.append(tf-ti)
                    err = [ spectral_error(S, Pi, Uk, Lambdask_halfinv, method=err_type) for Pi in Pis]
                    errors[graph_type][graph][cm].append(err)
                    timings[graph_type][graph][cm].append(times)
    
    results = {}
    results['graph_types'] = graph_types
    results['coarsening_methods'] = coarsening_methods
    results['rs'] = rs
    results['errors'] = errors
    results['timings'] = timings

    with open(saving_file_name, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    





    




