# scp elasalle@platinum1.cbp.ens-lyon.fr:/projects/users/elasalle/Parallel_Structured_Coarse_Grained_Spectral_Clustering/expes/parameters_influence/results_param_influence/influence_of_method_align3.pickle C:\Users\user\Documents\GitHub\Parallel_Structured_Coarse_Grained_Spectral_Clustering\expes\parameters_influence\results_param_influence\

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pickle
import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
    "font.size": 20
    # 'axes.titlesize': 15,
    # 'figure.titlesize': 20,
})

def plot_perf(nlist, err, color, label, perc=10, style='-', errorbar=True,
              alpha=0.2, lw=None, marker='o', markersize=10, ax=None, direction=-1):
    # print(err)
    if ax is not None:
        mean = np.mean(err, axis=direction)
        # print(mean)
        ax.plot(
            nlist,
            mean,
            linestyle=style,
            label=label,
            color=color,
            marker=marker,
            lw=lw,
            markersize=markersize)
        if errorbar:
            # print(np.percentile(err, perc, axis=direction))
            # print('\n')
            ax.fill_between(nlist, np.percentile(err, perc, axis=direction),
                            np.percentile(err, 100 - perc, axis=direction),
                            alpha=alpha, facecolor=color)
    else:
        plt.plot(
            nlist,
            err.mean(direction),
            linestyle=style,
            label=label,
            color=color,
            marker=marker,
            lw=lw,
            markersize=markersize)
        if errorbar:
            plt.fill_between(nlist, np.percentile(err, perc, axis=direction),
                             np.percentile(err, 100 - perc, axis=direction),
                             alpha=alpha, facecolor=color)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Test the influence of a parameter on the pasco performances.')
    parser.add_argument('-p', '--parameter', nargs='?', type=str,
                        help='parameter which influence is tested', default="rho",
                        choices=["rho", "n_tables", "method_align", "sampling"])
    parser.add_argument('-s', '--score', nargs='?', type=str,
                        help='score with which influence is tested', default="ami",
                        choices=["ami", "ari", "time", "modularity"])
    args = parser.parse_args()
    varying_param = args.parameter
    score = args.score

    # set directories
    res_dir = "results/"
    saving_file_name = res_dir + '/influence_of_' + varying_param + '.pickle'
    plots_dir = "../data/plots/param"

    # load results
    with open(saving_file_name, 'rb') as f:
        results = pickle.load(f)

    n = results["n"]
    k = results["k"]
    d = np.rint(results["avg_d"]/np.log(n)*2)/2
    alphas = results["alphas"]


    nx = len(alphas)
    nz = len(results["varying_values"])
    nrep = 10

    score_values = np.zeros((nx, nrep, nz))
    score_SC = np.zeros((nx, nrep))

    for ix in range(nx):
        for irep in range(nrep):
            score_SC[ix, irep] = results[ix][irep]["SC"][score]
            for iz in range(nz):
                score_values[ix, irep, iz] = results[ix][irep][iz][score]

    fig, ax = plt.subplots(1,1, figsize=(6,4), num=varying_param)
    perc = 20
    transparency = 0.2
    cm = colormaps.get_cmap("jet")
    labels_param = {
        "rho" : r"$\rho = $", 
        "n_tables" :  r"$t = $",
        "method_align" : "",
        "sampling" : ""
              }
    
    # plot thresholds
    plt.axvline(1/k, color='k', linestyle='--') # pasco threshold
    # plt.axvline((d-np.sqrt(d))/(d + (k-1)*np.sqrt(d)), color='k', linestyle=':') # SC threshold

    # plot SC
    plot_perf(alphas, score_SC, 'r', "SC", marker="D", perc=perc, style='--', alpha=transparency, ax=ax, direction=-1)

    #plot pasco
    for iz, val in enumerate(results["varying_values"]):
        color = cm(iz / nz)
        label = labels_param[varying_param] + str(val)
        plot_perf(alphas, score_values[:,:,iz], color, label, perc=perc, alpha=transparency, ax=ax, direction=-1)
    ax.grid()
    ax.set_ylabel(score)
    ax.set_xlabel(r"$\alpha$")
    if score=="time":
        ax.set_yscale('log')
    else:
        ax.set_ylim(-0.05,1.05)
    plt.tight_layout()
    legend = ax.legend(loc='right', bbox_to_anchor=(1, 0.5), bbox_transform=fig.transFigure, shadow=True)
    plt.subplots_adjust(right=0.8)  # Increase right margin


    figname = "influence_"+varying_param+"_"+score+"_n"+str(n)+"_k"+str(k)+"_d"+str(d)+".pdf"
    fig.savefig(plots_dir+figname)


    