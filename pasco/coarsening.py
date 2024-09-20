import warnings
import numpy as np
import networkx as nx
from scipy.sparse import csc_array, csr_array, coo_array, spdiags
import random
from sklearn import preprocessing
import scipy.sparse as sp
import multiprocessing as mp

#####################
# Custom exceptions #
#####################

class CoParameterError(Exception):
    pass
class SelfLoopError(Exception):
    pass
class GraphTypeError(Exception):
    pass


###############################################
# coarsening a graph given a coarsening table #
###############################################


def coarsen_current_graph(graph, table):
    # n_supernode is not even necessary
    G_prime = nx.Graph()
    G_prime.add_nodes_from([i for i in range(np.max(table))])
    for u, v, d in graph.edges(data=True):
        u_prime, v_prime = table[u], table[v]
        if (u_prime, v_prime) in G_prime.edges():
            G_prime[u_prime][v_prime]["weight"] += d["weight"]
        else:
            G_prime.add_edge(u_prime, v_prime, weight=d["weight"])
    return G_prime


def convert_table_to_array(table):
    target_size = np.max(table)+1
    graph_size = len(table)
    row = np.arange(graph_size)
    data = np.ones(graph_size)
    P = coo_array((data, (row, table)), shape=(graph_size, target_size))
    Pt, P = P.transpose().tocsr(), P.tocsr()
    return P, Pt


def coarsen_current_adj(A, P, Pt):
    target_A = Pt @ A @ P 
    ns = target_A.shape[0]
    diagonal = target_A.diagonal()
    return target_A - spdiags(0.5*diagonal, 0, ns, ns)


##################
# edge collapses #
##################


def _get_degree_from_cumdegs(cumdegs, i):
    if i == 0:
        return cumdegs[0]
    else:
        return cumdegs[i]-cumdegs[i-1]


def edge_collapse(graph, count_collapses, nb_collapses, batch_size):
    # remove the self loops by keeping G unmodified
    selfloops = [(u,v,d['weight']) for u,v,d in nx.selfloop_edges(graph, data=True)]
    graph.remove_edges_from([(u,v) for u,v,d in selfloops])
    cumdegs = np.cumsum([d for n, d in graph.degree()])  # degree are not taking into accounts weights
    local_table = np.arange(graph.number_of_nodes())  # initialize the current table
    while cumdegs[-1] > 0 and count_collapses < nb_collapses:
        us = random.choices(range(len(graph)), cum_weights=cumdegs, k=batch_size)
        if batch_size > 1:
            us = list(set(us))  # remove redundancies, not necessary if batch_size = 1
        for u in us:
            v = random.choice(list(graph.neighbors(u)))
            deg_u = _get_degree_from_cumdegs(cumdegs, u)
            deg_v = _get_degree_from_cumdegs(cumdegs, v)
            cumdegs[u:] -= deg_u
            cumdegs[v:] -= deg_v
            graph.remove_edge(u, v)
            if u == v:
                raise SelfLoopError("The sampled edge was a self loop and should not be.")
            # local_table[local_table == local_table[v]] = u
            local_table[u] = local_table[v]
            count_collapses += 1
    # restore the selfloops
    graph.add_weighted_edges_from(selfloops)
    return local_table, count_collapses


# using a scipy.sparse adjacency matrix
def get_incident_nodes_using_set(available_nodes, sampled_nodes, batch_size):
    us = []
    while len(us)<batch_size and len(available_nodes)>0:
            u = available_nodes.pop()
            if not(u in sampled_nodes):
                us.append(u)
    return us


def get_a_neighbor_adj(u, adj):
    neighbors = list(adj.indices[adj.indptr[u]:adj.indptr[u+1]])
    v = u
    while v == u:
        v = random.choice(neighbors)  # get a random neighbor of u that is not u
    return v


def edge_collapse_adj_uniformly(adj, count_collapses, nb_collapses, batch_size):
    graph_size = adj.shape[0]
    available_nodes = list(range(graph_size))
    random.shuffle(available_nodes)
    sampled_nodes = set()
    local_table = np.arange(graph_size)  # initialize the current table

    while count_collapses < nb_collapses:

        # get a list of nodes that have not been involved in collapses before
        us = get_incident_nodes_using_set(available_nodes, sampled_nodes, batch_size)
        if len(us) == 0:
            break
        for u in us:
            # sample a uniform neighbor of u (that is not u, in case of selfloops)
            v = get_a_neighbor_adj(u, adj)

            # update the sampled_nodes
            sampled_nodes.add(v) # if using set

            # update the assignment table according to the collapse of edge (u,v)
            local_table[u] = local_table[v]
            count_collapses += 1
    return local_table, count_collapses


# def get_degrees_adj(adj):
#     degs = adj.indptr[1:] - adj.indptr[:-1]
#     for i in range(adj.shape[0]):
#         if adj[i,i] != 0:
#             degs[i] -= 1
#     return degs

def get_degrees_adj(adj):
    diag_ = adj.diagonal()
    A = adj - spdiags(diag_, 0, adj.shape[0], adj.shape[0])
    return A.indptr[1:] - A.indptr[:-1]


def edge_collapse_adj_degree(adj, count_collapses, nb_collapses, batch_size):
    graph_size = adj.shape[0]
    cumdegs = np.cumsum(get_degrees_adj(adj))
    local_table = np.arange(graph_size) 

    while cumdegs[-1] > 0 and count_collapses < nb_collapses:
        us = random.choices(range(graph_size), cum_weights=cumdegs, k=batch_size)
        if batch_size > 1:
            us = list(set(us))  # remove redundancies, not necessary if batch_size = 1
        for u in us:
            # sample a uniform neighbor of u (that is not u, in case of selfloops)
            v = get_a_neighbor_adj(u, adj)

            # update the cumdegs
            deg_u = _get_degree_from_cumdegs(cumdegs, u)
            deg_v = _get_degree_from_cumdegs(cumdegs, v)
            cumdegs[u:] -= deg_u
            cumdegs[v:] -= deg_v
            
            # update the assignment table according to the collapse of edge (u,v)
            local_table[u] = local_table[v]
            count_collapses += 1
    return local_table, count_collapses


###################
# coarse graining #
###################


# an algo that does not start from the initial G at each iteration 
def update_final_table(final_table, current_table):
    return current_table[final_table]


def coarse_graining(G, ns, batch_size=1):
    le = preprocessing.LabelEncoder()
    graph_size = G.number_of_nodes()
    nb_collapses = graph_size-ns
    final_table = np.arange(graph_size)
    Gprime = G.copy()
    if not nx.is_weighted(Gprime):
        nx.set_edge_attributes(Gprime, 1, 'weight')  # turn the initial graph into a weighted graph.
    count_collapses = 0
    while count_collapses < nb_collapses:
        # compute the current table by collapsing some edges
        current_table, count_collapses = edge_collapse(Gprime, count_collapses, nb_collapses, batch_size)
        # clean the table so that it takes values from 0 to len(np.unique(current_table))
        current_table = le.fit_transform(current_table) 
        # update current graph and final hash-table
        Gprime = coarsen_current_graph(Gprime, current_table)
        final_table = update_final_table(final_table, current_table)
    return Gprime, final_table


# an algo based on scipy.sparse
def coarse_graining_adj(A, ns, batch_size=1, method='uniform_node_sampling'):
    """Coarse graining based on adjacency matrix as a scipy.sparse csr_array

    Parameters
    ----------
    A : scipy.sparse.csr_array
        adjacency matrix
    ns : int
        number of supernodes
    batch_size : int, optional
        number of edges to sample at a time, by default 1
    method : str, optional
        Sampling method used to sample the first incident node of the edge to collapse, by default 'uniform_node_sampling'

    Returns
    -------
    Aprime : same type as A
        adjacency matrix of the coarsened graph
    final_table : array_like
        The assignment table to the supernodes. Node i is assigned to supernode final_table[i].
    """
    le = preprocessing.LabelEncoder()
    graph_size = A.shape[0]
    nb_collapses = graph_size-ns
    final_table = np.arange(graph_size)  # csc format so that column slices are easy
    Aprime = A.copy()
    count_collapses = 0
    while count_collapses < nb_collapses:
        # compute the current table by collapsing some edges
        if method == 'uniform_node_sampling':
            current_table, count_collapses = edge_collapse_adj_uniformly(Aprime, count_collapses, nb_collapses, batch_size)
        elif method == 'degree_node_sampling':
            current_table, count_collapses = edge_collapse_adj_degree(Aprime, count_collapses, nb_collapses, batch_size)
        else:
            raise CoParameterError("The sampling method {} does not currently exist".format(method))
        # clean the table so that it takes values from 0 to len(np.unique(current_table))
        current_table = le.fit_transform(current_table)
        current_table = np.array(current_table, dtype=np.int32)
        # update current graph and final table
        P, Pt = convert_table_to_array(current_table)
        Aprime = coarsen_current_adj(Aprime, P, Pt)
        final_table = update_final_table(final_table, current_table)
    return Aprime, final_table


class Coarsening:
    def __init__(self, ns, batch_size, method_sampling='uniform_node_sampling'):
        """Coarsening class

        Parameters
        ----------
        ns : int
            number of supernodes
        batch_size : int, optional
            number of edges to sample at a time, by default 1
        method : str, optional
            Sampling method used to sample the first incident node of the edge to collapse, by default 'uniform_node_sampling'
        """
        self.ns = ns
        self.batch_size = batch_size
        self.method = method_sampling

    def fit_transform(self, graph):
        """Coarsen the graph.

        Parameters
        ----------
        graph : either networkx.Graph or scipy.sparse.csr_array
            A representation of the graph to coarsen

        Returns
        -------
        coarsened_graph : same type as graph
            The coarsened graph
        table : list
            The assignment table to the supernodes. Node i is assigned to supernode table[i]. 
        """
        if type(graph) == nx.Graph:
            graph_type = 'networkx'
            n = graph.number_of_nodes()
        elif type(graph) == sp.csr_array:
            graph_type = 'scipy_csr'
            n = graph.shape[0]
        else:
            raise GraphTypeError('Graph type is not correct. '
                      'Please use networkx.Graph or scipy.sparse.csr_array types')

        if n < self.ns:
            raise CoParameterError("n={} should be greater than ns={}.".format(n, self.ns))
        elif n == self.ns:
            return graph, np.arange(n)  # maybe .copy() is required

        if graph_type == 'networkx':
            if self.method == 'degree_node_sampling':
                return coarse_graining(graph, self.ns, self.batch_size)
            else:
                raise CoParameterError('Coarsening method {} is not implemented for networkx graph.'.format(self.method))
        elif graph_type == 'scipy_csr':
            return coarse_graining_adj(graph, self.ns, self.batch_size, self.method)

    def fit_transform_multiple(self, graph, n_tables, verbose=False, parallel=False):
        """Apply several time the random graph coarsening.

        Parameters
        ----------
        graph : either networkx.Graph or scipy.sparse.csr_array
            A representation of the graph to coarsen
        n_tables : int
            The number of coarsened graph desired
        parallel : bool, optional
            If True, the coarsening computations are done in parallel, by default True

        Returns
        -------
        coarsened_graphs : list of objects of the same type as graph
            The coarsened graphs
        tables : list of list
            The assignment tables to the supernodes.
        """        

        Gprimes = []
        tables = []
        if parallel:
            if verbose and n_tables == 1:
                warnings.warn('Using parallel coarsening for only one coarsened graph, this may be slower')
            args = (graph for _ in range(n_tables))
            cpu_count = min(mp.cpu_count(), n_tables)
            with mp.Pool(cpu_count) as pool_co:
                res = pool_co.map(self.fit_transform, args)
            Gprimes = [res[i][0] for i in range(n_tables)]
            tables = [res[i][1] for i in range(n_tables)]
            return Gprimes, tables
        else:
            if verbose and n_tables > 4:
                warnings.warn('For large number of coarsenings, consider using parallel=True')
            for _ in range(n_tables):
                Gprime, table = self.fit_transform(graph)
                Gprimes.append(Gprime)
                tables.append(table)
            return Gprimes, tables


















