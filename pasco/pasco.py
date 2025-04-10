from pasco.coarsening import Coarsening
from pasco.clustering import Clustering
from pasco.fusion import Fusion
from time import time

class Pasco:
    def __init__(self, k=None, rho=10, n_tables=10, batch_size=1, method_sampling="uniform_node_sampling", parallel_co=True,
                 solver="SC", parallel_cl=True, nb_align=10, method_align='ot',
                 method_fusion="hard_majority_vote", init_ref_method="auto", verbose=0, n_verbose=5, solver_args={}, optim_args={}):
        """_summary_

        Parameters
        ----------
        k : int or None, optional
            The number of communities. Must be provided if the clustering method used requires it, by default None
        rho : float, optional
            Compression factor. If the original graph has size N, the target size of the coarsened graph is floor(N/rho), by default 10
        n_tables : int, optional
            Number of repetition of coarsening, by default 10
        batch_size : int, optional
            Number of edges chosen at once, should be kept to 1, by default 1
        method_sampling : {"uniform_node_sampling", "degree_node_sampling"}, optional
            Sampling method used to sample the first incident node of the edge to collapse, by default "uniform_node_sampling"
        parallel_co : bool, optional
            Whether the coarsened graphs are computed in parallel, by default True
        solver :  {'SC', 'CSC', 'louvain', 'leiden', 'graclus', 'MDL', 'infomap'} or callable, optional
            Clustering algorithm to apply on the coarsened graphs, by default "SC"
        parallel_cl : bool, optional
            Whether the clustering of the coarsenined graphs are computed in parallel, by default True
        nb_align : int, optional
            Maximum of iterations in the alignment of the obtained partitions, by default 10
        method_align : {"lin_reg", "many_to_one", "ot"}, optional
            name of the alignment method, by default 'many_to_one'
        method_fusion : str, optional
            name of the fusion method, by default 'majority_vote'
        init_ref_method : {"auto", "first_partition", "max_clusters", "min_clusters", "random", "max_mod"}, optional
            method to initialize the reference, by default "first_partition"
        verbose : int, optional
            Verbose level, by default 0
        n_verbose : int, optional
            Number of iterations between prints in the alignment/fusion part, by default 5
        solver_args : dict, optional
           Arguments to pass to the solver functions, by default {}
        optim_args : dict, optional
            Extra arguments to pass to the Fusion function of PASCO, by default {}
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


