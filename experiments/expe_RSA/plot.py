import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
    "font.size": 20
    # 'axes.titlesize': 15,
    # 'figure.titlesize': 20,
})

if __name__ == '__main__':

    plot_dir = "../data/plots/RSA"
    fig_name = "RSA_both_graphs"

    res_dir = "results/"
    saving_file_name = res_dir + '/results' + '.pickle'

    # load results
    with open(saving_file_name, 'rb') as f:
        results = pickle.load(f)

    graph_types = results['graph_types']
    coarsening_methods = results['coarsening_methods']
    rs = results['rs']
    errors = results['errors']
    timings = results['timings']

    labels = {"pasco":"PASCO",
              'variation_edges':'variation_edges',
              'heavy_edge':'heavy_edge'}

    perc = 20
    alpha = 0.2
    tab20 = colormaps.get_cmap('tab10')
    lsty = ['-', '--', ':']
    mkr = ['o', '^', 'P']

    nb_graph_types = len(graph_types)
    nb_max_graphs = max([len(errors[graph_type]) for graph_type in graph_types])
    fig, axss = plt.subplots(nb_graph_types,nb_max_graphs, figsize=(5*nb_max_graphs, 6*nb_graph_types), sharey="row", squeeze=False)

    for iax, (axs, graph_type) in enumerate(zip(axss, graph_types)):
        for jax, (ax, graph) in enumerate(zip(axs, errors[graph_type].keys())):
            for i, cm in enumerate(coarsening_methods):
                    j=0 #only one error type
                    errs = errors[graph_type][graph][cm]
                    ax.plot(rs, np.mean(errs, axis=1), color=tab20(i), marker=mkr[j], linestyle=lsty[j], label=labels[cm])
                    ax.fill_between(rs, np.percentile(errs, perc, axis=1), np.percentile(errs, 100 - perc, axis=1), alpha=alpha, facecolor=tab20(i))
            ax.set_yscale('log')
            ax.grid()
            ax.set_xlabel(r"compression rate $(1-\rho^{-1})$")
            if jax==0:
                ax.set_ylabel("RSA")
            if graph_type=="SBM":
                ax.set_title(r"$k = {}$, $\alpha = {}$".format(graph[1], graph[3]))
            else:
                ax.set_title(graph)
    # plt.subplots_adjust(bottom=0.25, hspace=0.2)  # Increase bottom margin
    legend_height = 0.04
    plt.tight_layout(pad=1.5, h_pad=2.5, rect=(0,legend_height*2,1,1))
    plt.legend(loc='center', bbox_to_anchor=(0.5, legend_height), bbox_transform=fig.transFigure, shadow=True, ncol=len(coarsening_methods))
    fig.savefig(plot_dir+"/"+fig_name+".pdf")



    fig_name = "timings_both_graphs"
    
    nb_graph_types = len(graph_types)
    nb_max_graphs = max([len(timings[graph_type]) for graph_type in graph_types])
    fig, axss = plt.subplots(nb_graph_types,nb_max_graphs, figsize=(5*nb_max_graphs, 6*nb_graph_types), sharey="row", squeeze=False)

    for iax, (axs, graph_type) in enumerate(zip(axss, graph_types)):
        for jax, (ax, graph) in enumerate(zip(axs, timings[graph_type].keys())):
            for i, cm in enumerate(coarsening_methods):
                    j=0 #only one error type
                    times = timings[graph_type][graph][cm]
                    ax.plot(rs, np.mean(times, axis=1), color=tab20(i), marker=mkr[j], linestyle=lsty[j], label=labels[cm])
                    ax.fill_between(rs, np.percentile(times, perc, axis=1), np.percentile(times, 100 - perc, axis=1), alpha=alpha, facecolor=tab20(i))
            ax.set_yscale('log')
            ax.grid()
            ax.set_xlabel(r"compression rate $(1-\rho^{-1})$")
            if jax==0:
                ax.set_ylabel("time")
            if graph_type=="SBM":
                ax.set_title(r"$k = {}$, $\alpha = {}$".format(graph[1], graph[3]))
            else:
                ax.set_title(graph)
    # plt.subplots_adjust(bottom=0.25, hspace=0.2)  # Increase bottom margin
    legend_height = 0.04
    plt.tight_layout(pad=1.5, h_pad=2.5, rect=(0,legend_height*2,1,1))
    plt.legend(loc='center', bbox_to_anchor=(0.5, legend_height), bbox_transform=fig.transFigure, shadow=True, ncol=len(coarsening_methods))
    fig.savefig(plot_dir+"/"+fig_name+".pdf")