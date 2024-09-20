import numpy as np
from scipy.sparse import coo_array, csr_array
from sklearn.preprocessing import LabelEncoder
import ot
import time


class FuParameterError(Exception):
    pass


def many_to_one_alignment(ref, partition, use_sparse_matrices=True, to_log=False):
    """Align a (hard-)partition with respect to a reference (soft-)partition assuming they both have k clusters using the many-to-one method.

    Parameters
    ----------
    ref : array_type of shape (n, output_nb_clusters)
        The reference partition. Might be a soft-partition with entries in [0,1].
        It satisfies: ref.sum(axis=1)=1.
    partition : array_type of shape (n,)
        The partition to align w.r.t. ref. The node i is assigned to cluster partition[i]
    use_sparse_matrices: bool, optional
        Whether to use sparse matrix multiplications to compute the alignement. Preferrable when k i small compared to n, default True.

    Returns
    -------
    aligned_partition : array_type of shape (n, output_nb_clusters)
        The aligned parition, where output_nb_clusters is the number of clusters in the reference.
        It is a priori a hard-clustering with entries in {0,1}.
    """
    output_nb_clusters = ref.shape[1]
    n = ref.shape[0]
    k = len(np.unique(partition))
    if to_log:
        log_ = {}
    if use_sparse_matrices:
        P = coo_array((np.ones(n), (np.arange(n), partition)), shape=(n, k))
        st = time.time()
        C = coo_array((np.ones(k), (np.arange(k), (P.transpose()
                      @ ref).argmax(1))), shape=(k, output_nb_clusters))
        ed = time.time()
        aligned_partition = (P @ C).todense()  # why todense ?
        # aligned_partition = P @ C
    else:
        st = time.time()
        C = np.zeros((k, output_nb_clusters))
        for i in range(n):
            C[partition[i], :] += ref[i, :]
        index_clusters = C.argmax(axis=1)[partition]
        ed = time.time()
        # np.add.at(C, partition, ref)
        aligned_partition = np.zeros((n, k))
        aligned_partition[np.arange(n), index_clusters] = 1
    if to_log:
        log_['time'] = ed - st
        log_['cost'] = np.linalg.norm(ref - aligned_partition)**2
        return aligned_partition, log_
    else:
        return aligned_partition


def ot_alignment(ref, partition, to_log=False):
    """Align a (hard-)partition with respect to a reference (soft-)partition using optimal transport.

    Parameters
    ----------
    ref : array_type of shape (n, output_nb_clusters)
        The reference partition. Might be a soft-partition with entries in [0,1].
        It satisfies: ref.sum(axis=1)=1.
    partition : array_type of shape (n,)
        The partition to align w.r.t. ref. The node i is assigned to cluster partition[i]
    use_weights: bool, optional

    Returns
    -------
    aligned_partition : array_type of shape (n, output_nb_clusters)
        The aligned parition, where output_nb_clusters is the number of clusters in the reference.
        It is a priori a soft-clustering with entries in [0,1].
    """
    output_nb_clusters = ref.shape[1]
    n = ref.shape[0]
    k = len(np.unique(partition))
    P = coo_array((np.ones(n), (np.arange(n), partition)), shape=(n, k))
    C = - P.transpose() @ ref  # C is k times output_nb_clusters

    # uniform weights
    a = np.ones(k) / k
    b = np.ones(output_nb_clusters) / output_nb_clusters  
    
    st = time.time()
    T = ot.emd(a, b, C)
    ed = time.time()
    # we multiply by k in order to have {0, 1} entries, but should not have
    # an impact MAYBE TO REMOVE
    # if k == output_nb_clusters:
    #     aligned_partition = P @ (k * T)
    # else:
    aligned_partition = P @ T

    if to_log:
        log = {}
        log['T'] = T
        log['time'] = ed - st
        log['cost'] = np.linalg.norm(ref - aligned_partition)**2
        return aligned_partition, log

    else:
        return aligned_partition


def lin_reg_alignment(ref, partition, use_sparse_matrices=True, to_log=False):
    """Align a (hard-)partition with respect to a reference (soft-)partition assuming they both have k clusters using the linear regression method.

    Parameters
    ----------
    ref : array_type of shape (n, output_nb_clusters)
        The reference partition. Might be a soft-partition with entries in [0,1].
        It satisfies: ref.sum(axis=1)=1.
    partition : array_type of shape (n,)
        The partition to align w.r.t. ref. The node i is assigned to cluster partition[i]
    use_sparse_matrices: bool, optional
        Whether to use sparse matrix multiplications to compute the alignement. Preferrable when k i small compared to n, default True.

    Returns
    -------
    aligned_partition : array_type of shape (n, output_nb_clusters).
        The aligned parition, where output_nb_clusters is the number of clusters in the reference.
        The aligned parition. It is a priori a soft-clustering with entries in [0,1].
    """
    output_nb_clusters = ref.shape[1]
    n = ref.shape[0]
    k = len(np.unique(partition))
    if to_log:
        log_ = {}
    if use_sparse_matrices:
        P = coo_array((np.ones(n), (np.arange(n), partition)), shape=(n, k))
        st = time.time()
        C = P.transpose() @ ref / np.array(P.sum(0)).reshape((-1, 1))
        ed = time.time()
        aligned_partition = P @ C
    else:
        st = time.time()
        C = np.zeros((k, output_nb_clusters))
        cluster_sizes = np.zeros(k)  # to test maybe it is output_nb_clusters
        for i in range(n):
            C[partition[i], :] += ref[i, :]
            cluster_sizes[partition[i]] += 1
        C = C / cluster_sizes.reshape((-1, 1))
        ed = time.time()
        aligned_partition = C[partition, :]
    if to_log:
        log_['time'] = ed - st
        log_['cost'] = np.linalg.norm(ref - aligned_partition)**2
        return aligned_partition, log_
    else:
        return aligned_partition


def align_and_fuse(partitions, ref, method_alignment, method_fusion="majority_vote"):
    """Performs the alignment of every partition into a reference ref
    then performs a majority vote, i.e., a average partition from the input partitions

     Parameters
     ----------
    partitions : list of hard-partition of shape (n,)
        the partitions to align with the reference
    n : int
        the number of objects
    ref : array_like of shape (n, output_nb_clusters)
        A reference partition.
    method_alignment : func
        The alignment method to use
     method_fusion : str, optional
         fusion method use to combined the aligned partitions, by default "majority_vote".

     Returns
     -------
     fused_partition : numpy.array of shape (n, output_nb_clusters)
         the average partition
     """
    n, output_nb_clusters = ref.shape
    n_tables = len(partitions)
    fused_partition = np.zeros((n, output_nb_clusters))
    aligned_partitions = []

    for i, partition in enumerate(partitions):
        aligned_partition = method_alignment(ref, partition)
        fused_partition += aligned_partition  # will become the mean
        aligned_partitions.append(aligned_partition)
    fused_partition /= n_tables  # now, this is similar to np.mean

    if method_fusion == "majority_vote" or method_fusion == "soft_majority_vote":
        return fused_partition, aligned_partitions

    elif method_fusion == "hard_majority_vote":
        fused_partition_hard = np.zeros((n, output_nb_clusters))
        fused_partition_hard[np.arange(n), fused_partition.argmax(axis=1)] = 1
        return fused_partition_hard, aligned_partitions

    else:
        raise FuParameterError(
            "The fusion method {} does not exist.".format(method_fusion))


def voting(partitions):
    partition = partitions[0]
    return partition


def find_best_partition_modularity(partitions, A):
    """
    Returns the index of the partition with highest modularity given an adjacency matrix A
    :param partition: list of list, partitions of the graph
    :param A: csr array, adjacency matrix
    :return: index of the partition with highest modularity
    """

    mods = np.zeros(len(partitions))
    n = A.shape[0]
    deg = A.sum(1)  # degree
    m = A.sum() / 2  # number of edges
    normalizing_constant = (1.0 / (2 * m))
    L = normalizing_constant * (A - np.outer(deg, deg) * normalizing_constant)
    for k, partition in enumerate(partitions):
        G = np.zeros((n, len(np.unique(partition))))
        G[np.arange(n), partition] = 1
        mods[k] = np.sum((L.T @ G) * G)
    best_p = np.argmax(mods)

    return best_p


class Fusion:
    def __init__(self,
                 output_nb_clusters=None,
                 nb_align=10,
                 method_align='many_to_one',
                 method_fusion='majority_vote',
                 init_ref_method="auto",
                 tol=1e-5,
                 verbose=False,
                 n_verbose=5,
                 log=False,
                 break_when_tol_attained=True,
                 true_graph=None,
                 numItermax=None,
                 stopThr=None):
        """Combine several clustering by aligning them and fusing them to output one clustering.

        Parameters
        ----------
        output_nb_clusters : int
            the number of desired clusters after fusion, ignored if "first_partition" or "max_mod" is used.
        nb_align : int, optional
            number of maximal alignment iteration to perform, should be at least 1, default is 10.
        method_align : {"lin_reg", "many_to_one", "ot"}, optional
            name of the alignment method, by default 'many_to_one'
        method_fusion : str, optional
            name of the fusion method, by default 'majority_vote'
        init_ref_method : {"auto", "first_partition", "max_clusters", "min_clusters", "random", "max_mod"}, optional
            method to initialize the reference, by default "first_partition"
        tol : float, optional
            Tolerance on the iterates below which the algorithm stops.
        verbose : bool, optional
            Verbosity, by default False.
        n_verbose : int, optional
            number of iterations before print, default is 10. Ignored if verbose is False.
        log : bool, optional
            Whether to store the results in a dict, by default False.
        break_when_tol_attainted : bool, optional
            Whether to stop when the tolerance is attained, by default True.
        true_graph: csr_array, optional
            The adjency matrix of the true graph that we want to partition, default None.
            Required for init_ref_method == max_mod.
        """
        self.output_nb_clusters = output_nb_clusters
        self.max_iter = nb_align
        self.method_align = method_align
        self.method_fusion = method_fusion
        self.init_ref_method = init_ref_method
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.break_when_tol_attained = break_when_tol_attained
        self.true_graph = true_graph
        self.n_verbose = n_verbose

        if self.init_ref_method == "max_mod":
            if self.true_graph is None:
                raise FuParameterError(
                    "init_ref_method == max_mod requires defining true_graph")
            if self.true_graph is not None and not isinstance(
                    self.true_graph, csr_array):
                raise TypeError(
                    "init_ref_method == max_mod requires the true_graph as a csr sparse array")
        if log:
            self.results = dict()
            self.results['references'] = []
            self.results['aligned_partitions'] = []
        if self.max_iter < 1:
            raise FuParameterError(
                "The number of alignment iteration should be at least 1, got {}".format(
                    self.max_iter))

        possible_alignment_methods = [
            "lin_reg", "many_to_one", "ot"]

        if self.method_align == "lin_reg" and (
                self.method_fusion == "soft_majority_vote" or self.method_fusion == "majority_vote"):
            raise FuParameterError(
                "The alignment method 'lin_reg' should not be used with the fusion method 'soft_majority_vote'")

        if self.method_align == "lin_reg":
            self.alignment_method = lin_reg_alignment
        elif self.method_align == "many_to_one":
            self.alignment_method = many_to_one_alignment
        elif self.method_align == "ot":
            self.alignment_method = lambda ref, partition: ot_alignment(
                ref, partition, use_weights=False)
        else:
            raise FuParameterError(
                " The alignment method {} does not exist, it should be in {}.".format(
                    self.method_align, possible_alignment_methods))

        possible_init_methods = [
            "auto",
            "first_partition",
            "max_clusters",
            "min_clusters",
            "random",
            "max_mod"]
        if self.init_ref_method not in possible_init_methods:
            raise FuParameterError(
                "The method {} to initialize the reference does not exist, it should be in {}".format(
                    self.init_ref_method, possible_init_methods))
        if self.output_nb_clusters is None and self.init_ref_method == "random":
            raise FuParameterError(
                "The value of the output_nb_clusters can not be None with 'random' initialization."
            )

    def fit_transform(self, partitions):
        """Compute one clustering that agrees the most with all the given clusterings after alignment.

        Parameters
        ----------
        partitions : list of array_like of shape (n,)
            The list of each cluster assignments.

        Returns
        -------
        final_partition : numpy.arrray for shape (n,)
            The custer assignments of the final clustering.
        """
        n = len(partitions[0])

        if len(partitions) == 1:
            return partitions[0]

        if self.init_ref_method == "auto":
            k_partitions = [len(np.unique(partition))
                            for partition in partitions]
            all_same_nb_of_cluster = len(set(k_partitions)) == 1
            if all_same_nb_of_cluster or self.output_nb_clusters is None:
                # as in "first_partition"
                self.output_nb_clusters = len(np.unique(partitions[0]))
                ref = np.zeros((n, self.output_nb_clusters))
                ref[np.arange(n), partitions[0]] = 1
            elif np.median(k_partitions) > self.output_nb_clusters:
                # as in "min_clusters"
                self.output_nb_clusters = min(k_partitions)
                ref = np.zeros((n, self.output_nb_clusters))
                ref[np.arange(n), partitions[np.argmin(k_partitions)]] = 1
            else:
                # as in "max_clusters"
                self.output_nb_clusters = max(k_partitions)
                ref = np.zeros((n, self.output_nb_clusters))
                ref[np.arange(n), partitions[np.argmax(k_partitions)]] = 1
        elif self.init_ref_method == "first_partition":  # a discuter
            # the ref is set to the first partition but turned into an
            # numpy.array of shape (n, output_nb_clusters)
            self.output_nb_clusters = len(np.unique(partitions[0]))
            ref = np.zeros((n, self.output_nb_clusters))
            ref[np.arange(n), partitions[0]] = 1
        elif self.init_ref_method == "max_clusters":
            k_partitions = [len(np.unique(partition))
                            for partition in partitions]
            self.output_nb_clusters = max(k_partitions)
            ref = np.zeros((n, self.output_nb_clusters))
            ref[np.arange(n), partitions[np.argmax(k_partitions)]] = 1
        elif self.init_ref_method == "min_clusters":
            k_partitions = [len(np.unique(partition))
                            for partition in partitions]
            self.output_nb_clusters = min(k_partitions)
            ref = np.zeros((n, self.output_nb_clusters))
            ref[np.arange(n), partitions[np.argmin(k_partitions)]] = 1
        elif self.init_ref_method == "random":
            # the ref is set to a random partition
            ref = np.zeros((n, self.output_nb_clusters))
            ref[np.arange(n), np.random.randint(
                0, self.output_nb_clusters, n)] = 1
        elif self.init_ref_method == "max_mod":
            p = find_best_partition_modularity(partitions, self.true_graph)
            # the ref is set to the best partition in terms of modularity but turned into an
            # numpy.array of shape (n, output_nb_clusters)
            if self.verbose:
                print(f"The best partition is partition number {p}")
            ref = np.zeros((n, len(np.unique(partitions[p]))))
            ref[np.arange(n), partitions[p]] = 1

        if self.log:
            self.results['references'].append(ref)
        loop = True
        i = 0
        losses = []

        def loss(ref, realigned_partitions):
            return np.mean([((ref - realigned_partition)**2).sum() for realigned_partition in realigned_partitions])

        while loop:
            ref_new, realigned_partitions = align_and_fuse(
                partitions, ref, method_alignment=self.alignment_method, method_fusion=self.method_fusion)

            losses.append(loss(ref_new, realigned_partitions))

            if self.verbose and (i % self.n_verbose == 0):
                print(
                    f'-- iter {i} --- loss : {float(losses[-1]): .7e}')

            criteria = False
            if i >= 2:
                if losses[-1] == 0.:
                    criteria = True
                else:
                    abs_loss = abs(losses[-1]-losses[-2])
                    relative_loss = abs_loss / losses[-1]
                    # sufficient decreases
                    criteria = True if (relative_loss < self.tol) else False
                    if self.verbose and (i % self.n_verbose == 0):
                        print(f'            --- rel loss : {float(relative_loss): .7e}')
            if criteria and self.break_when_tol_attained:
                if self.verbose:
                    print(
                        '---------- relative_loss below threshold: stop algorithm ----------')
                loop = False

            i += 1
            if i == self.max_iter:
                if self.verbose:
                    print('---------- max_iter attained ----------')
                loop = False
            ref = ref_new
            if self.log:
                self.results['references'].append(ref)
                # self.results['aligned_partitions'].append(aligned_partitions)

        if self.log:
            self.results['loss'] = losses
        final_partition = np.argmax(ref, axis=1)
        # ensure that the output clusters are numbered between 0 and some k-1
        final_partition = LabelEncoder().fit_transform(final_partition)
        return final_partition
