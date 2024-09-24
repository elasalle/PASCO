import networkx as nx
import os
import numpy as np
import scipy.sparse as sp


class GraphParameterError(Exception):
    pass


class ConnectednessError(Exception):
    pass


class NoGraphError(Exception):
    pass


def generate_SBM(k, n_k, pin, pout, n_tries_for_connectedness, seed):
    is_connected = False
    n_tries = 0
    while not is_connected and n_tries < n_tries_for_connectedness:
        G = nx.planted_partition_graph(k, n_k, pin, pout, seed+n_tries)
        is_connected = nx.is_connected(G)
        n_tries += 1
    return G


def generate_or_import_SBM(n, k, pin, pout, data_folder="data/", n_tries_for_connectedness=5, seed=None):

    n_k = n//k

    if n < k:
        raise GraphParameterError("n should be greater than k. Received n={} and k={}".format(n,k))
    if pout <= 0 or pin < pout or pin > 1:
        raise GraphParameterError("pin and pout should satisfy  0 < pout <= pin <= 1. Received pin={} and pout={}".format(pin, pout))
    
    graph_file_name = "G_n{}_k{}_pin{}_pout{}_seed{}".format(n, k, pin, pout, seed)
    graph_path = data_folder+graph_file_name

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if os.path.exists(graph_path):
        G = nx.read_adjlist(graph_path, nodetype=int)
        print("Graph imported")
    else:
        print("Graph currently does not exist. We generate it.")
        G = generate_SBM(k, n_k, pin, pout, n_tries_for_connectedness, seed)
        nx.write_adjlist(G, graph_path)
        print("Graph generated")

    if not nx.is_connected(G):
        raise ConnectednessError("The graph is not connected.")
    return G


def read_real_datasets(data_folder='graphs/', dataset='arxiv'):
    """
    Return the adjacency matrix as a csr_array of a graph
    :param data_folder: location of the data folder
    :param dataset: name of the graph to read
    :return: Adjacency matrix of the graph and labels associated to nodes
    """
    try:
        A = sp.load_npz(data_folder + dataset + '_adjacency_lcc.npz')
        ground_truth_partition = np.load(data_folder + dataset + '_node_label_lcc.npz')
        partition_true = ground_truth_partition['arr_0']
    except:
        print("Processing dataset. This may take some time. This will be done only once.")
        A = sp.load_npz(data_folder + dataset + '_adjacency.npz')
        ground_truth_partition = np.load(data_folder + dataset + '_node_label.npz')
        G = nx.from_scipy_sparse_array(A)

        if not nx.is_connected(G):
            print("The graph is not connected. Computation will be run on the largest connected component.")
            # get largest conected component if graph is not connected
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            n = len(G.nodes())
            mapping = dict(zip(G, range(n)))
            nodelist = list(G.nodes())
            G = nx.relabel_nodes(G, mapping)
            partition_true = ground_truth_partition['arr_0'][nodelist]
            A = nx.adjacency_matrix(G, nodelist=range(n))
        else:
            partition_true = ground_truth_partition['arr_0']

        sp.save_npz(data_folder + dataset + '_adjacency_lcc.npz', A)
        np.savez(data_folder + dataset + '_node_label_lcc.npz', np.array(partition_true))
    return A, partition_true

