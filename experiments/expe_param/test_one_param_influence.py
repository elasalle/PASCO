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
    parser.add_argument('-p', '--parameter', nargs='?', type=str,
                        help='parameter which influence is tested', default="rho",
                        choices=["rho", "n_tables", "method_align"])
    parser.add_argument('-n', '--graphsize', nargs='?', type=int,
                        help='gaph size', default=10**4)
    parser.add_argument('-k', '--nbclusters', nargs='?', type=int,
                        help='number of communities', default=20)
    parser.add_argument('-d', '--density', nargs='?', type=float,
                        help='density coefficient', default=1.5)
    args = parser.parse_args()

    # experiment parameters
    nb_repetitions = 10
    varying_param = args.parameter

    # default pasco parameters
    pasco_param = {
        "rho" : 10,
        "n_tables" : 10,  # number of coarsened graphs,
        "sampling" : "uniform_node_sampling",
        "solver" : 'SC', # lobpcg is for spectral clustering.,
        "assign_labels" : 'cluster_qr',
        "method_align" : 'ot',
        "method_fusion" : 'hard_majority_vote',
        "batch_size" : 1
    }
    varying_values = {
        "rho" : [1.5,3,5,10,15],
        "n_tables" : [3,5,10,15],
        "method_align" : ["lin_reg", "many_to_one", "ot"],
        "sampling" : ["uniform_node_sampling", "degree_node_sampling"]
    }

    # SBM parameters
    n = args.graphsize
    k = args.nbclusters
    d = args.density
    avg_d = d*np.log(n)
    n_alphas = 10
    n_k = n//k
    # alpha_c = (avg_d - np.sqrt(avg_d)) / (avg_d + (k-1)*np.sqrt(avg_d))
    alpha_c = 1/(k-1)
    alphas = np.linspace(0, 1.33*alpha_c , n_alphas+1)[1:] # avoid    alpha=0 which yields disconnected SBM graphs
    pins = avg_d / ((1 + (k-1) *alphas )*n_k)
    pouts = alphas * pins 
    partition_true = np.array([i for i in range(k) for j in range(n_k)])

    # save some useful parameters
    results = {
        "varying_param" : varying_param,
        "varying_values" : varying_values[varying_param],
        "true_partition" : partition_true,
        "n" : n,
        "k" : k,
        "avg_d" : avg_d,
        "alphas" : alphas
    }

    # set directories
    res_dir = "results/"
    saving_file_name = res_dir + '/influence_of_' + varying_param  + '.pickle'
    # check that the results folder does exist, if not, create it.
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # IMPORT OR GENERATE THE GRAPH
    for expe_i, (pin, pout) in enumerate(zip(pins, pouts)):
        results[expe_i] = {}
        print("graph setting : {} / {}".format(expe_i+1, len(pins)))
        for rep_i in range(nb_repetitions):
            print("  repetition : {} / {}".format(rep_i+1, nb_repetitions))
            G = generate_or_import_SBM(n, k, pin, pout, data_folder="../data/graphs/SBMs/", seed=2024+100*rep_i)
            A = nx.adjacency_matrix(G , nodelist=range(n))

            results[expe_i][rep_i] = {}
            results[expe_i][rep_i]["true_modularity"] = modularity_graph(G, partition_true)

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
            results[expe_i][rep_i]["SC"]["ami"] = ami(partition_true, partition_sc)
            results[expe_i][rep_i]["SC"]["ari"] = ari(partition_true, partition_sc)
            results[expe_i][rep_i]["SC"]["modularity"] = modularity_graph(G, partition_sc)
            results[expe_i][rep_i]["SC"]["nb_clusters"] = len(np.unique(partition_sc))

            with open(saving_file_name, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #################
            # compute pasco #
            #################
            print("    pasco computations start")
            for var_i, varying_val in enumerate(varying_values[varying_param]):
                print("      param : {} / {}".format(var_i+1, len(varying_values[varying_param])))
                pasco_param[varying_param] = varying_val
                results[expe_i][rep_i][var_i] = {}

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
                results[expe_i][rep_i][var_i]["ami"] = ami(partition_true, partition_pasco)
                results[expe_i][rep_i][var_i]["ari"] = ari(partition_true, partition_pasco)
                results[expe_i][rep_i][var_i]["modularity"] = modularity_graph(G, partition_pasco)
                results[expe_i][rep_i][var_i]["nb_clusters"] = len(np.unique(partition_pasco))

                with open(saving_file_name, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("test finished. Results saved in {}".format(saving_file_name))

            