import os
os.environ["OMP_NUM_THREADS"] = "1"

from pasco.pasco import Pasco
from pasco.data_generation import generate_or_import_SBM
import numpy as np
import networkx as nx
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.preprocessing import LabelEncoder    
from time import time

import sys
sys.path.append('/projects/users/elasalle/PASCO/')
from graclus.graclus_wrapper import run_graclus

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # generate graph
    n = int(1e5) # number of nodes 
    k = 100  # number of communities 
    d = 1.5 # to set the average degree
    alpha = 1/(2*(k-1)) # ratio of probabilities. Here half the conjectured threshold. See Paper. 

    n_k = n//k # number of nodes per community
    avg_d = d*np.log(n) # average degree
    pin = avg_d / ((1 + (k-1) *alpha )*n_k) # inside community edge probability
    pout = alpha * pin # between communities edge probability
    print(pin, pout)


    partition_true = np.array([i for i in range(k) for j in range(n_k)]) # the true nodes partition
    G = generate_or_import_SBM(n, k, pin, pout, data_folder="experiments/data/graphs/SBMs/", seed=2024)
    A = nx.adjacency_matrix(G , nodelist=range(n))

    #compute pasco
    rho = 10 # reduction factor (the coarsened graph will have a size rho times smaller)
    R = 10 # number of repetitions of the coarsening. R should be kept below the number of CPUs so that all clusterings can be computed in one batch.
    solver = "SC" # we use SC to compute the partition of the coarsened graphs.

    ti = time()
    pasco = Pasco(k, rho, R, solver=solver)
    partition_pasco = pasco.fit_transform(A)
    tf = time()

    print("AMI with PASCO+SC : {:5.3f}".format(ami(partition_pasco, partition_true)))
    print("Computation time: {:5.3f}sec \n".format((tf-ti)))

    # compute graclus
    ti  = time()
    clusters = run_graclus(G,k)
    partition_graclus = np.array(list(clusters.values()))
    tf = time()

    print("AMI with graclus : {:5.3f}".format(ami(partition_graclus, partition_true)))
    print("Computation time: {:5.3f}sec \n".format((tf-ti)))

