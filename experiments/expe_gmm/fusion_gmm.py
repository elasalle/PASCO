# %% Fusion of GMM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pasco.fusion import Fusion
import random
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from pasco.utils import SimpleClustering, plot_perf
import time

cmap = plt.cm.get_cmap('tab10')
# %%
choice_metric = {'ARI': adjusted_rand_score, 'AMI': adjusted_mutual_info_score}

nb_samples = [500, 400, 200]
n = np.sum(nb_samples)
centers = np.array([[0, 1],
                    [1, 0],
                    [0, 0]])

X, y = make_blobs(n_samples=nb_samples, centers=centers,
                  cluster_std=0.25, shuffle=False)
# %%
nt_total = 30
all_clustering = np.zeros((nt_total, n))
ari_two = np.zeros((nt_total, ))
ari_three = np.zeros((nt_total, ))

# rho = 0.95  # taux de reduction
# m = round((1-rho)*n)
# print('Number of total samples={}, number of reduced samples {}'.format(n, m))
# K = 2
for i in range(nt_total):
    K = np.random.randint(2, 15)
    # K = 3
    # while K == 3:
    # K = np.random.randint(2, 7)
    # K = 10
    sp = SimpleClustering(K=K, random_state=None)
    # kmeans = KMeans(K)
    # idx = random.sample(range(n), m)
    # XX = X[idx]
    # kmeans.fit(XX)
    sp.fit(X, y)
    labels = sp.predict(X)
    all_clustering[i] = labels
    ari_two[i] = adjusted_rand_score(labels, y)
    # ari_three[i] = adjusted_rand_score(labels, y)

# %%
n_repet = 5
output_nb_clusters = 3

fusion_methods = [
    'ot',
    'quadratic_ot',
    # 'ot_weights',
    # 'ot_ref_weights',
    # 'ot_partition_weights',
    'lin_reg',
    'many_to_one'
]
res = {}
res_timings = {}
all_partitions = {}

for nt in range(nt_total):
    for j in range(n_repet):
        for method in fusion_methods:
            print('Method = {}'.format(method))
            if method not in res:
                res[method] = np.zeros((nt_total, n_repet))
                res_timings[method] = np.zeros((nt_total, n_repet))
            st = time.time()
            fusion = Fusion(output_nb_clusters=output_nb_clusters,
                            nb_align=10,
                            method_align=method,
                            init_ref_method='random',
                            method_fusion='hard_majority_vote',
                            log=False,
                            verbose=False)
            ed = time.time()
            partitions = [all_clustering[i] for i in range(nt + 1)]
            realigned_partition = fusion.fit_transform(partitions)
            all_partitions[method, nt, j] = realigned_partition
            res_timings[method][nt, j] = ed - st
# %%
for l, metric_to_show in enumerate(['ARI', 'AMI']):
    for nt in range(nt_total):
        for j in range(n_repet):
            for method in fusion_methods:
                res[method][nt, j] = choice_metric[metric_to_show](y,
                                                                   all_partitions[method, nt, j])

# %%
fs = 19

metric_to_show = 'AMI'
method_to_show = 'quadratic_ot'
names_to_plot = {'ot': 'lin-ot', 'quadratic_ot': 'quad-ot',
                 'lin_reg': 'lin-reg', 'many_to_one': 'many-to-one'}
xx = range(1, nt_total + 1)
perc = 10

fig, ax = plt.subplots(1, 5, figsize=(22, 5), layout="constrained")
ax[0].scatter(X[:, 0], X[:, 1], c=[cmap(y[i]) for i in range(len(y))], alpha=1,
              edgecolor='k')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Dataset', fontsize=fs+2)
for i in range(1, 3):
    colors = [cmap(int(label)) for label in all_clustering[i]]
    ax[i].scatter(X[:, 0], X[:, 1], c=colors, alpha=1,
                  edgecolor='k')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(
        'Partition n°{}/{}'.format(i, nt_total), fontsize=fs+2)

fused_partition = all_partitions[method_to_show, nt_total-1, 0]
perf = choice_metric[metric_to_show](
    y, fused_partition)
ax[i+1].scatter(X[:, 0], X[:, 1],
                c=[cmap(fused_partition[i])
                   for i in range(len(fused_partition))],
                alpha=1,
                edgecolor='k')
ax[i+1].set_xticks([])
ax[i+1].set_yticks([])
ax[i+1].set_title('Fused partition ({2}) \n {1} = {0:.3f}'.format(
    perf, metric_to_show, names_to_plot[method_to_show]), fontsize=fs+2)

to_show = ['ot',
           'lin_reg',
           'many_to_one',
           'quadratic_ot']

for c, method in enumerate(to_show):

    metric_fusion = res[method]
    plot_perf(
        xx,
        metric_fusion,
        color=cmap(c),
        label=names_to_plot[method],
        ax=ax[4],
        direction=1,
        markersize=7,
        lw=3,
        perc=perc,
        alpha=0.15,
        std=True,
        coeff=1.96)
kmeans = KMeans(3)
labels = kmeans.fit_predict(X)

metric_full = choice_metric[metric_to_show](y, labels)
ax[4].hlines(
    y=metric_full,
    xmin=0,
    xmax=nt_total,
    lw=4,
    color='k',
    linestyle='dashed',
    label='k-means')
ax[4].grid()
ax[4].set_ylim([0.1, 1.05])
ax[4].set_xlabel('number of partitions $t$', fontsize=fs)
ax[4].set_ylabel('{}'.format(metric_to_show), fontsize=fs)
ax[4].set_xlim([1, nt_total + 1])
ax[4].tick_params(axis='both', which='major', labelsize=fs - 2)
ax[4].tick_params(axis='both', which='minor', labelsize=fs - 2)

ax[4].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
             ncol=2, fancybox=False, shadow=True,
             fontsize=14)
plt.savefig('../paper/illustration/example_fusion.pdf')
# %%
fs = 13
to_show = ['ot', 'lin_reg', 'many_to_one', 'quadratic_ot']
fig, ax = plt.subplots(figsize=(10, 4))
timings = []
colors = []
err = []
for c, method in enumerate(to_show):
    timings.append(res_timings[method].mean())
    colors.append(cmap(c))
    err.append(res_timings[method].std())
ax.bar(to_show, timings, yerr=err, color=colors)
ax.set_yscale('log')
ax.set_ylabel('time (in sec)', fontsize=fs)
ax.grid()
ax.tick_params(axis='both', which='major', labelsize=fs)
ax.tick_params(axis='both', which='minor', labelsize=fs)
fig.suptitle('Averaged time fusion', fontsize=fs+2)
# %%


# %%
# fs = 18

# metric_to_show = 'ARI'
# method_to_show = 'ot'

# fig, ax = plt.subplots(1, 5, figsize=(20, 4))
# ax[0].scatter(X[:, 0], X[:, 1], c=[cmap(y[i]) for i in range(len(y))], alpha=1,
#               edgecolor='k')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# ax[0].set_title('True dataset', fontsize=fs+2)
# for i in range(1, 4):
#     colors = [cmap(int(label)) for label in all_clustering[i]]
#     ax[i].scatter(X[:, 0], X[:, 1], c=colors, alpha=1,
#                   edgecolor='k')
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
#     ax[i].set_title(
#         'Partition n°{}/{}'.format(i, nt_total), fontsize=fs+2)

# fused_partition = all_partitions[method_to_show, nt_total-1, 0]
# perf = choice_metric[metric_to_show](
#     y, fused_partition)
# ax[i+1].scatter(X[:, 0], X[:, 1],
#                 c=[cmap(fused_partition[i])
#                    for i in range(len(fused_partition))],
#                 alpha=1,
#                 edgecolor='k')
# ax[i+1].set_xticks([])
# ax[i+1].set_yticks([])
# ax[i+1].set_title('Fused partition ({2}) \n {1} = {0:.3f}'.format(
#     perf, metric_to_show, method_to_show), fontsize=fs+2)
# plt.tight_layout()
# %%
# fs = 20

# fig, ax = plt.subplots(1, 2, figsize=(15, 7))
# xx = range(1, nt_total + 1)
# perc = 10

# for l, metric_to_show in enumerate(['ARI', 'AMI']):
#     axx = ax[l]
#     for nt in range(nt_total):
#         for j in range(n_repet):
#             for method in fusion_methods:
#                 res[method][nt, j] = choice_metric[metric_to_show](y,
#                                                                    all_partitions[method, nt, j])

#     to_show = ['ot', 'lin_reg', 'many_to_one', 'quadratic_ot']
#     for c, method in enumerate(to_show):

#         metric_fusion = res[method]
#         plot_perf(
#             xx,
#             metric_fusion,
#             color=cmap(c),
#             label=method,
#             ax=axx,
#             direction=1,
#             markersize=7,
#             lw=2,
#             perc=perc,
#             alpha=0.15,
#             std=True,
#             coeff=1.96)
#     kmeans = KMeans(3)
#     labels = kmeans.fit_predict(X)
#     metric_full = choice_metric[metric_to_show](y, labels)
#     # ax[3].hlines(
#     #     y=ari_two.mean(),
#     #     xmin=0,
#     #     xmax=nt_total,
#     #     lw=3,
#     #     color=cmap(c + 1),
#     #     linestyle='dashed',
#     #     label='avg. k-means on subsets')
#     axx.hlines(
#         y=metric_full,
#         xmin=0,
#         xmax=nt_total,
#         lw=4,
#         color='k',
#         linestyle='dashed',
#         label='k-means')
#     # ax.fill_between(xx, np.min(ari_two), np.max(ari_two),
#     #                 alpha=0.2, facecolor=cmap(c + 1))
#     axx.grid()
#     axx.set_ylim([0.1, 1.05])
#     if l == 1:
#         axx.legend(loc='lower right', fontsize=fs)
#     axx.set_xlabel('number of partitions $t$', fontsize=fs)
#     axx.set_ylabel('{}'.format(metric_to_show), fontsize=fs)
#     # axx.set_xticks(range(1, nt_total+1))
#     axx.set_xlim([1, nt_total + 1])
#     axx.tick_params(axis='both', which='major', labelsize=fs - 2)
#     axx.tick_params(axis='both', which='minor', labelsize=fs - 2)
# fig.suptitle('Fusion on GMM', fontsize=fs+2)
# plt.tight_layout()
