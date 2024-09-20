import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder


def plot_perf(nlist, err, color, label, perc=10, style='-', errorbar=True,
              alpha=0.2, lw=None, marker='o', markersize=10, ax=None,
              direction=0, std=False, coeff=1.96):
    ax.plot(
        nlist,
        err.mean(direction),
        style,
        label=label,
        color=color,
        marker=marker,
        lw=lw,
        markersize=markersize)
    if errorbar:
        if std:
            N = err.shape[direction]
            ax.fill_between(nlist, err.mean(direction) +
                            (coeff/np.sqrt(N))*np.std(err, axis=direction),
                            err.mean(direction)-(coeff/np.sqrt(N)) *
                            np.std(err, axis=direction),
                            alpha=alpha, facecolor=color)
        else:
            ax.fill_between(nlist, np.percentile(err, perc, axis=direction),
                            np.percentile(err, 100 - perc, axis=direction),
                            alpha=alpha, facecolor=color)


class SimpleClustering():
    # Nearest neighbour clustering
    def __init__(self, K=1, metric='euclidean', random_state=None):
        self.metric = metric
        self.K = K
        np.random.seed(random_state)

    def any_all(self, target, data):
        return np.any(np.all(target == data, axis=1))

    def fit(self, X, y=None):
        assert self.K <= X.shape[0]

        # draw K random pivotal points
        if y is None:
            idx = np.random.randint(0, X.shape[0], self.K)
            pivotal_point = X[idx]
            self.pivotal_point = pivotal_point
        else:
            all_classes = np.unique(y)
            nb_classes = len(all_classes)
            points_chosen = []
            sigma = np.random.permutation(all_classes)
            k = 0
            while k < min(self.K, nb_classes):  # draw min(self.K, nb_classes) points
                # draw one random point in class cl
                Xsubsampled = X[y == sigma[k]]
                idx = np.random.randint(0, Xsubsampled.shape[0])
                points_chosen.append(Xsubsampled[idx])
                k += 1
            np_points_chosen = np.array(points_chosen)
            # if there is more points to draw take them uniformly amoung X
            # (expect these which are already chosen)
            left_to_choose = self.K - nb_classes
            while left_to_choose > 0:
                idx = np.random.randint(0, X.shape[0])
                # if already chosen choose another one
                while self.any_all(X[idx], np_points_chosen):
                    idx = np.random.randint(0, X.shape[0])
                np_points_chosen = np.vstack([np_points_chosen, X[idx]])
                left_to_choose -= 1
            self.pivotal_point = np_points_chosen
        return self

    def fit_transform(self, X, y=None):
        # X is n times d
        # y is k times d
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        D = cdist(X, self.pivotal_point, metric=self.metric)
        partition = LabelEncoder().fit_transform(np.argmin(D, axis=1)).astype(int)
        return partition
