import os
os.environ["OMP_NUM_THREADS"] = "1"

import networkx as nx
from pasco.clustering_tools import spectral_clustering
from pasco.pasco import Pasco
from pasco.data_generation import generate_or_import_SBM
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari
from time import time
import pickle
import argparse
import datetime
from dateutil import tz
import warnings
warnings.filterwarnings("ignore")


def modularity_graph(G, clustering):
    k = len(np.unique(clustering))
    # print(np.max(clustering), k, len(clustering))
    partition = [set() for _ in range(k)]
    for i in range(G.number_of_nodes()):
        partition[clustering[i]].add(i)
    m = nx.community.modularity(G, partition)
    return m


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Test the influence of a parameter on the pasco performances.')
    parser.add_argument('-n', '--graphsize', nargs='?', type=int,
                        help='gaph size', default=10**5)
    parser.add_argument('-d', '--density', nargs='?', type=float,
                        help='density coefficient', default=1.5)
    args = parser.parse_args()

    # experiment parameters
    nb_repetitions = 10

    # default pasco parameters
    pasco_param = {
        "rho" : 10,
        "n_tables" : 10,  # number of coarsened graphs,
        "sampling" : "uniform_node_sampling",
        "solver" : 'SC', 
        "assign_labels" : 'cluster_qr',
        "method_align" : 'ot',
        "method_fusion" : 'hard_majority_vote',
    }

    ks = np.array([20,100,1000])
    rhos = [3,5,10,15]
    
    # SBM parameters
    n = args.graphsize
    d = args.density
    avg_d = d*np.log(n)

    # save some useful parameters
    results = {
        "n" : n,
        "ks" : ks,
        "rhos": rhos,
        "avg_d" : avg_d,
        "nrep" : nb_repetitions
    }

    # set directories
    res_dir = "results/"
    saving_file_name = res_dir + 'timings' + '.pickle'

    # IMPORT OR GENERATE THE GRAPH
    for expe_i, k in enumerate(ks):
        results[expe_i] = {}
        print("graph setting : {} / {}".format(expe_i+1, len(ks)))
        # compute graph parameters
        n_k = n//k
        alpha = 1/(2*(k-1))
        pin = avg_d / ((1 + (k-1) *alpha )*n_k)
        pout = alpha * pin
        partition_true = np.array([i for i in range(k) for j in range(n_k)])
        for rep_i in range(nb_repetitions):
            print("  repetition : {} / {}".format(rep_i+1, nb_repetitions))
            G = generate_or_import_SBM(n, k, pin, pout, data_folder="../data/graphs/SBMs/", seed=2024+100*rep_i)
            A = nx.adjacency_matrix(G , nodelist=range(n))

            results[expe_i][rep_i] = {}

            ##############
            # compute SC #
            ##############
            print("    SC computations start")
            start_sc = time()
            partition_sc = spectral_clustering(A, k)
            stop_sc = time()
            # set the computational time
            results[expe_i][rep_i]["SC"] = {}
            results[expe_i][rep_i]["SC"]["time"] = stop_sc - start_sc
            results[expe_i][rep_i]["SC"]["partition"] = partition_sc
            results[expe_i][rep_i]["SC"]["nb_clusters"] = len(np.unique(partition_sc))
            results[expe_i][rep_i]["SC"]["ami"] = ami(partition_true, partition_sc)

            with open(saving_file_name, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #################
            # compute pasco #
            #################
            print("    pasco computations start")
            for var_i, rho in enumerate(rhos):
                print("      param : {} / {}".format(var_i+1, len(rhos)))
                results[expe_i][rep_i][var_i] = {}

                pasco_param["rho"] = rho
                pasco_param["n_tables"] = rho

                ns = n // pasco_param["rho"]

                # pasco
                if pasco_param["n_tables"] > 1:
                    parallel_coarsening = True
                    parallel_clustering = True
                else:
                    parallel_coarsening = False
                    parallel_clustering = False
                pasco_clustering = Pasco(k=k, rho=pasco_param["rho"], n_tables=pasco_param["n_tables"], parallel_co=parallel_coarsening,
                                         solver=pasco_param["solver"], parallel_cl=parallel_clustering, 
                                         nb_align=50, method_align=pasco_param["method_align"], method_fusion=pasco_param["method_fusion"], 
                                         verbose=True)
                partition_pasco, timings = pasco_clustering.fit_transform(A, return_timings=True)
                print("Pasco pipeline done.")

                # set the computational time
                results[expe_i][rep_i][var_i]["coarsening"] = timings["coarsening"]
                results[expe_i][rep_i][var_i]["clustering"] = timings["clustering"]
                results[expe_i][rep_i][var_i]["fusion"] = timings["fusion"]
                results[expe_i][rep_i][var_i]["time"] = timings["coarsening"] + timings["clustering"] + timings["fusion"]
                results[expe_i][rep_i][var_i]["partition"] = partition_pasco
                results[expe_i][rep_i][var_i]["ami"] = ami(partition_true, partition_pasco)
                results[expe_i][rep_i][var_i]["nb_clusters"] = len(np.unique(partition_pasco))

                with open(saving_file_name, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            