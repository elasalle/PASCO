from pasco.coarsening import Coarsening
from pasco.clustering import Clustering
from pasco.fusion import Fusion
from time import time

class Pasco:
    def __init__(self, k=None, rho=10, n_tables=10, batch_size=1, method_sampling="uniform_node_sampling", parallel_co=True,
                 solver="SC", parallel_cl=True, nb_align=10, method_align='ot',
                 method_fusion="hard_majority_vote", init_ref_method="auto", verbose=0, n_verbose=5, solver_args={}, optim_args={}):
        """
        Initialization of the class Pasco
        :param k: number of clusters (if known). This is ignored if not used by the solver
        :param rho: reduction parameter = number of nodes divided by number of supernodes
        :param n_tables: number of coarsened graphs
        :param batch_size: number of edges to merge at a time
        :param method_sampling: method to sample the edges to be merged
        :param parallel: whether to use parallel computation when it's possible
        :param solver: solver to use
        :param assign_labels: method to assign labels
        :param nb_align: number of alignments in the fusion process
        :param method_align: method to align the partitions
        :param method_fusion: method to fuse the partitions after alignment
        :param init_ref_method: reference method for alignment
        :param verbose: whether to print or not
        """
        self.k = k
        self.rho = rho
        self.n_tables = n_tables
        self.batch_size = batch_size
        self.method_sampling = method_sampling
        self.parallel_co = parallel_co
        self.solver = solver
        self.parallel_cl = parallel_cl
        self.nb_align = nb_align
        self.method_align = method_align
        self.method_fusion = method_fusion
        self.init_ref_method = init_ref_method
        self.verbose = verbose
        self.n_verbose = n_verbose
        self.solver_args = solver_args
        self.optim_args = optim_args

    def fit_transform(self, A, return_timings=False):
        """
        To be written
        :param A: adjacency matrix
        :return: partition of the graph using pasco
        """

        # Coarsening
        n = A.shape[0]  # number of nodes
        ns = int(n/self.rho)  # number of supernodes
        timings = {}

        # Coarsening
        t_co_i = time()
        coarsener = Coarsening(ns, self.batch_size, method_sampling=self.method_sampling)
        Gprimes, tables = coarsener.fit_transform_multiple(A, self.n_tables, parallel=self.parallel_co)
        t_co_f = time()
        timings["coarsening"] = t_co_f - t_co_i
        if self.verbose:
            print("Coarsening done")

        # Clustering
        t_cl_i = time()
        clusterer = Clustering(self.k, tables, solver=self.solver, parallel=self.parallel_cl, **self.solver_args)
        partitions = clusterer.fit_transform(Gprimes)
        t_cl_f = time()
        timings["clustering"] = t_cl_f - t_cl_i
        if self.verbose:
            print("Clustering done")

        # Alignment & Fusion
        t_fu_i = time()
        fuser = Fusion(self.k, nb_align=self.nb_align, method_align=self.method_align, method_fusion=self.method_fusion,
                       init_ref_method=self.init_ref_method, true_graph=A, verbose=self.verbose, n_verbose=self.n_verbose, **self.optim_args)
        final_partition = fuser.fit_transform(partitions)
        t_fu_f = time()
        timings["fusion"] = t_fu_f - t_fu_i
        if self.verbose:
            print("Fusion done")

        if return_timings:
            return final_partition, timings
        else:
            return final_partition


