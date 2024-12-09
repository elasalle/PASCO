from sklearn.cluster import SpectralClustering


def my_clustering(A, k):
    clusterer = SpectralClustering(n_clusters=k, affinity='precomputed', eigen_solver='lobpcg', assign_labels='cluster_qr')
    partition = clusterer.fit_predict(A)
    return partition

def my_clustering2(A, k, eigen_solver):
    clusterer = SpectralClustering(n_clusters=k, affinity='precomputed', eigen_solver=eigen_solver, assign_labels='cluster_qr')
    partition = clusterer.fit_predict(A)
    return partition