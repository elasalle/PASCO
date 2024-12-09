import os
os.environ["OMP_NUM_THREADS"] = "1"

import networkx as nx
from pasco.pasco import Pasco
from pasco.data_generation import generate_or_import_SBM
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari
import multiprocessing
from time import time

if __name__ == '__main__':

    n = int(1e5) # number of nodes 
    k = 100  # number of communities 
    d = 1.5 # to set the average degree
    alpha = 1/(2*(k-1)) # ration of probabilities. Here half the conjectured threshold. See Paper. 

    n_k = n//k # number of nodes per community
    avg_d = d*np.log(n) # average degree
    pin = avg_d / ((1 + (k-1) *alpha )*n_k) # inside community edge probability
    pout = alpha * pin # between communities edge probability

    partition_true = np.array([i for i in range(k) for j in range(n_k)]) # the true nodes partition
    G = generate_or_import_SBM(n, k, pin, pout, data_folder="experiments/data/graphs/SBMs/", seed=2024)
    A = nx.adjacency_matrix(G , nodelist=range(n))



    ti = time()
    clusterer = SpectralClustering(n_clusters=k, affinity='precomputed', eigen_solver='lobpcg', assign_labels='cluster_qr')
    partition_SC = clusterer.fit_predict(A)
    tf = time()

    print("AMI with SC : {:5.3f}".format(ami(partition_SC, partition_true)))
    print("Computation time: {:5.3f}sec".format((tf-ti)))



    n_cpu = multiprocessing.cpu_count()

    rho = 5 # reduction factor (the coarsened graph will have a size rho times smaller)
    R = min(rho, n_cpu) # number of repetitions of the coarsening. We keep R below the number of cpu so that all clusterings can be computed in one batch.
    solver = "SC" # we use SC to compute the partition of the coarsened graphs.

    ti = time()
    pasco = Pasco(k, rho, R, solver=solver)
    partition_pasco, t = pasco.fit_transform(A, return_timings=True)
    tf = time()

    print(t)
    print("AMI with PASCO+SC : {:5.3f}".format(ami(partition_pasco, partition_true)))
    print("Computation time: {:5.3f}sec".format((tf-ti)))