import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import networkx as nx
from pasco.pasco import Pasco
from pasco.data_generation import read_real_datasets
import  pasco.clustering_tools as ct
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari
import pickle
import json
import argparse
import datetime
from dateutil import tz
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test on real data. To evaluate time and performance of various methods including pasco.')
    parser.add_argument('-d', '--dataset', nargs='?', type=str,
                        help='dataset on which the experiment is performed', default="arxiv",
                        choices=["arxiv", "mag", "products"])
    args = parser.parse_args()

    # hyper-parameters
    res_dir = "results/"
    rhos = [1,10]
    ns_tables = [1,3,5,10,15]  # number of coarsened graphs
    solvers = ['infomap', 'SC','leiden', 'louvain', 'CSC', 'MDL' ]  # possible solvers are {'SC', 'graclus', 'louvain', 'leiden', 'CSC', 'MDL', 'infomap'}
    dataset = args.dataset

    # load graph
    dataset = args.dataset  # can be 'mag', 'arxiv', 'products'
    data_folder = '../graphs/' + dataset + '/'
    A, partition_true = read_real_datasets(data_folder=data_folder, dataset=dataset)
    print("Graph {} imported".format(dataset))

    # global hyper-parameters
    n = A.shape[0]
    k = np.unique(np.array(partition_true)).shape[0]
    method_align = 'ot'
    # method_align = 'quadratic_ot'
    method_fusion = 'hard_majority_vote'
    init_ref_method = "auto"

    sorted_solvers = solvers.copy()
    sorted_solvers.sort()
    
    suffix = method_align
    saving_file_name = res_dir + '/res_' + dataset + '_' + '_'.join(sorted_solvers) + "_" + suffix +'.pickle' # when using pickle to save the results
    # saving_file_name = res_dir + '/res_' + dataset + '_' + '_'.join(sorted_solvers) + "_" + suffix + '.json' # when using json to save the results
    # saving_file_name = res_dir + '/res_' + dataset + '_' + '_'.join(sorted_solvers) + '.npy'

    results = {"true_partition" : list(partition_true)}
    results["true_modularity"] = ct.modularity_graph(A, partition_true)
    results["true_gnCut"] = ct.generalized_normalized_cut(A, partition_true)
    results["true_dl"] = ct.description_length(A, partition_true)
    print("Quality of the true partition computed. \n")

    for solver in solvers:
        results[solver] = {}
        for rho in rhos:
            results[solver][rho] = {}
            for n_tables in ns_tables:
                if n_tables > 1 and rho == 1:
                    continue
                results[solver][rho][n_tables] = {}

                ns = n // rho

                print("Solver: " + solver)
                print("  Compression factor: ", rho)
                print("  Number of tables: ", n_tables)
                print(datetime.datetime.now(tz.tzlocal()).strftime('  %Y/%m/%d %H:%M:%S'))

                # pasco
                if n_tables > 1:
                    parallel_coarsening = True
                    parallel_clustering = True
                else:
                    parallel_coarsening = False
                    parallel_clustering = False
                pasco_clustering = Pasco(k=k, rho=rho, n_tables=n_tables, parallel_co=parallel_coarsening,
                              solver=solver, parallel_cl=parallel_clustering, 
                             method_align=method_align, method_fusion=method_fusion, init_ref_method=init_ref_method,
                             verbose=True)
                partition_pasco, timings = pasco_clustering.fit_transform(A, return_timings=True)
                print("Pasco pipeline done.")

                # set the computational time
                results[solver][rho][n_tables]["coarsening"] = timings["coarsening"]
                results[solver][rho][n_tables]["clustering"] = timings["clustering"]
                results[solver][rho][n_tables]["fusion"] = timings["fusion"]
                results[solver][rho][n_tables]["time"] = timings["coarsening"] + timings["clustering"] + timings["fusion"]
                # results[solver][rho][n_tables]["ideal_time"] = timings["coarsening"] + timings["clustering"]/n_tables + timings["fusion"] # not relevant anymore

                # compute the nmi score
                results[solver][rho][n_tables]["partition"] = list(partition_pasco)
                results[solver][rho][n_tables]["nmi"] = nmi(partition_true, partition_pasco)
                results[solver][rho][n_tables]["ami"] = ami(partition_true, partition_pasco)
                results[solver][rho][n_tables]["ari"] = ari(partition_true, partition_pasco)
                results[solver][rho][n_tables]["modularity"] = ct.modularity_graph(A, partition_pasco)
                results[solver][rho][n_tables]["gnCut"] = ct.generalized_normalized_cut(A, partition_pasco)
                results[solver][rho][n_tables]["dl"] = ct.description_length(A, partition_pasco)
                results[solver][rho][n_tables]["nb_clusters"] = len(np.unique(partition_pasco))
                print("Quality of the estimated partition computed. \n")

        with open(saving_file_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(saving_file_name, 'wb') as f:
        #     json.dump(results, f)

        # np.save(saving_file_name, results) 