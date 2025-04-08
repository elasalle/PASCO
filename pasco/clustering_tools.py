from  importlib.util import find_spec
gt_exists = find_spec("graph_tool") is not None
if gt_exists:
    import graph_tool.all as gt
im_exists = find_spec("infomap") is not None
if im_exists:
    from infomap import Infomap
graclus_exists = find_spec("torch_cluster")
if graclus_exists:
    import torch
    from torch_cluster import graclus_cluster
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder    
from sklearn.cluster import SpectralClustering
from CSC.compressed_spectral_clustering import CSC
from sknetwork.clustering import Louvain

import leidenalg as la
import igraph as ig





#########################
# Clustering Algorithms #
#########################


class ClGraphTypeError(Exception):
    pass


def _ckeck_graph_type(G):
    if type(G) == nx.Graph:
        adj = nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
    elif type(G) == sp.csr_array:
        adj = G
    else:
        raise ClGraphTypeError('Graph type {} is not correct. '
                  'Please use networkx.Graph or scipy.sparse.csr_array types'.format(type(G)))
    return adj

def spectral_clustering(G, k):
    adj = _ckeck_graph_type(G)
    clusterer = SpectralClustering(n_clusters=k, affinity='precomputed',
                                   eigen_solver='lobpcg', assign_labels='cluster_qr')
    partition = clusterer.fit_predict(adj)
    return partition


def compressiveSC(G, k, param={"solver":"cgs"}):
    adj = _ckeck_graph_type(G)
    dict_G = {
        'W': adj,
        'k': k,
        'N': adj.shape[0]
        }
    C_est, _, timings, _, _, _ = CSC(dict_G, param, verbose=False)
    partition = np.argmax(C_est, axis=1)
    return partition


def louvain_clustering(G, resolution=1.0):
    adj = _ckeck_graph_type(G)
    louvain = Louvain(resolution=resolution, return_probs=False, return_aggregate=False, sort_clusters=False)
    return louvain.fit_predict(sp.csr_matrix(adj))


def leiden_clustering(G):
    adj = _ckeck_graph_type(G)
    iG = ig.Graph.Weighted_Adjacency(sp.coo_matrix(adj), mode="undirected")
    partition = la.find_partition(iG, la.ModularityVertexPartition, weights='weight').membership
    partition = LabelEncoder().fit_transform(np.array(partition))
    return partition

if graclus_exists:
    def graclus_clustering(G):
        adj = _ckeck_graph_type(G)
        Acoo = adj.tocoo()
        row = torch.tensor(Acoo.row, dtype=torch.int64)
        col = torch.tensor(Acoo.col, dtype=torch.int64)
        weight = torch.tensor(Acoo.data)
        cluster = graclus_cluster(row, col, weight)
        partition = LabelEncoder().fit_transform(cluster.numpy())
        return partition


if gt_exists:
    def MDL(G, k=None):
        adj = _ckeck_graph_type(G)
        
        g = gt.Graph(adj, directed=False)
        state = gt.minimize_blockmodel_dl(g, 
                                        state_args=dict(recs=[g.ep.weight], rec_types=["discrete-binomial"], B=k),
                                        multilevel_mcmc_args=dict(B_min=k, B_max=k)) 
        # state.multiflip_mcmc_sweep(beta=np.inf, niter=100)
        b = state.get_blocks()
        partition = np.array([b[i] for i in range(adj.shape[0])])
        partition = LabelEncoder().fit_transform(partition)
        return partition


if im_exists:
    def infomap_clustering(G, k=None):
        partition = []
        im = Infomap(two_level=True, silent=True, preferred_number_of_modules=k, inner_parallelization=False)

        # links = []
        # for i, j in zip(*G.nonzero()):
        #     links.append((i, j))
        # im.add_links(links)

        if type(G) == sp.csr_array:
            nxG = nx.from_scipy_sparse_array(G)  # can be improved with im.add_links((1, 2), (1, 3))
        elif type(G) == nx.Graph:
            nxG = G
        else:
            raise ClGraphTypeError('Graph type {} is not correct. '
                      'Please use networkx.Graph or scipy.sparse.csr_array types'.format(type(G)))
        
        im.add_networkx_graph(nxG)
        im.run()
        for node, module in im.modules:
            partition.append(module-1)  # modules are labeled from 1 to n
        del im
        return partition



################################
# measuring clustering quality #
################################


def generalized_normalized_cut(A, partition):
    n = A.shape[0]
    k = len(np.unique(partition))
    P = sp.coo_array((np.ones(n), (np.arange(n), partition)), shape=(n, k)).tocsc()
    all_cuts = P.T @ A @ P
    cuts = all_cuts.diagonal()
    norms = all_cuts.sum(axis=0)
    return np.mean(cuts/norms)


def modularity_graph(A, partition):
    G = nx.from_scipy_sparse_array(A)
    k = len(np.unique(partition))
    partition_nx = [set() for _ in range(k)]
    for i in range(G.number_of_nodes()):
        partition_nx[partition[i]].add(i)
    m = nx.community.modularity(G, partition_nx)
    return m


if gt_exists:
    def description_length(A, partition):
        g = gt.Graph(A, directed=False)
        b = g.new_vp("int")
        b.fa = partition
        bs = gt.BlockState(g, b)
        return bs.entropy()
