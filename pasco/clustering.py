from pasco import clustering_tools as ct
import multiprocessing as mp

# import dask
# from dask.distributed import Client, LocalCluster, wait


class BadParameterError(Exception):
    pass


def coarsened_to_full(coarsened_clustering, table, n):
    partition = [coarsened_clustering[table[i]]
                 for i in range(n)]  # a bit dirty to clean
    return partition


class Clustering:
    def __init__(self, k=None, tables=[[]], solver='SC', parallel=False, **kwargs):
        """A class to perform spectral clustering on several coarsened graphs

        Parameters
        ----------
        k : int or None, optional
            number of desired clusters, if required by the clustering solver.
            If None it is ignored and not passed to the solver, else the value is passed to the "k" argument of the solver.
            By default None
        tables : list of lists
            The list of the assignments to the supernodes in the coarsened graphs.
        solver : {'SC', 'CSC', 'louvain', 'leiden', 'graclus', 'MDL', 'infomap'} or callable, optional
            solver used to compute the clustering of the coarsened graphs, by default 'SC'.
        parallel : boolean, optional
            whether the clustering computations on the coarsened graphs should be done in parallel, by default False.
        **kwargs : dict,
            keywords arguments to pass to the clustering solver if it is a callable.
        """
        self.k = k
        self.tables = tables
        self.n = len(tables[0])  # number of nodes in the original graph
        self.n_tables = len(tables)
        self.parallel = parallel
        self.solver = solver
        self.solver_args = kwargs
        possible_solvers = ['SC', 'graclus', 'louvain',
                            'leiden', 'CSC', 'MDL', 'infomap']
        if isinstance(self.solver, str):
            self.is_str = True
            if self.solver not in possible_solvers:
                raise BadParameterError('found solver = {} but possible solvers are {}'.format(
                    self.solver, possible_solvers))
        elif callable(self.solver):
            self.is_str = False
        else:
            raise BadParameterError(
                'solver should be either a string or a callable, received {}'.format(type(self.solver)))

        if self.is_str:
            if self.solver == 'graclus':
                self.fc = ct.graclus_clustering
            elif self.solver == 'louvain':
                self.fc = ct.louvain_clustering
            elif self.solver == 'leiden':
                self.fc = ct.leiden_clustering
            elif self.solver == 'SC':
                self.fc = ct.spectral_clustering
                self.solver_args["k"] = self.k
            elif self.solver == 'CSC':
                self.fc = ct.compressiveSC
                self.solver_args["k"] = self.k
            elif self.solver == 'MDL':
                self.fc = ct.MDL
                self.solver_args["k"] = self.k
            elif self.solver == 'infomap':
                self.fc = ct.infomap_clustering
                if self.k is not None:  # k can be given to infomap
                    self.solver_args["k"] = self.k
        else:
            self.fc = self.solver
            if self.k is not None:
                self.solver_args["k"] = self.k

    def _apply_clustering(self, graph):
        return self.fc(graph, **self.solver_args)

    def fit_transform(self, graphs):
        """Compute the spectral clustering of the coarsened graphs

        Parameters
        ----------
        graphs : list of either networkx.Graph or scipy.sparse.csr_array
            The coarsened graphs.

        Returns
        -------
        partitions : list of list
            The clusterings of each coarsened graph.
        """

        if self.parallel:
            cpu_count = min(mp.cpu_count(), self.n_tables)
            with mp.Pool(cpu_count) as pool:
                # res_async = pool.map_async(self._apply_clustering, graphs)
                # coarsened_clusterings = res_async.get()
                coarsened_clusterings = pool.map(self._apply_clustering, graphs)
                args = ((coarsened_clusterings[i], self.tables[i], self.n) for i in range(
                    self.n_tables))
                partitions = pool.starmap(coarsened_to_full, args)
            return partitions
        else:
            partitions = []
            for graph, table in zip(graphs, self.tables):
                coarsened_clusterings = self._apply_clustering(graph)
                partition = coarsened_to_full(
                    coarsened_clusterings, table, self.n)
                partitions.append(partition)
            return partitions
