# scp elasalle@platinum1.cbp.ens-lyon.fr:/projects/users/elasalle/Parallel_Structured_Coarse_Grained_Spectral_Clustering/expes/parameters_influence/results_param_influence/timings_ot.pickle C:\Users\user\Documents\GitHub\Parallel_Structured_Coarse_Grained_Spectral_Clustering\expes\parameters_influence\results_param_influence\

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
    description='Test the influence of k pasco performances.')
    parser.add_argument('-s', '--score', nargs='?', type=str,
                        help='score with which influence is tested', default="time",
                        choices=["ami", "time"])
    parser.add_argument('-r', '--relative', type=bool, default=False)
    args = parser.parse_args()
    score = args.score
    relative = args.relative

    # set directories
    res_dir = "results/"
    saving_file_name = res_dir + 'timings' + '.pickle'
    plots_dir = "../data/plots/param"

    # load results
    with open(saving_file_name, 'rb') as f:
        results = pickle.load(f)

    n = results["n"]
    ks = results["ks"]
    d = np.rint(results["avg_d"]/np.log(n)*2)/2
    rhos = [3,5,10,15] 
    # rhos = results["rhos"] # for the next run


    nx = len(rhos)
    nz = len(ks)
    nrep = 10
    # nrep = results["nrep"]

    score_values = np.zeros((nx, nrep, nz))
    score_SC = np.zeros((nrep, nz))

    for iz, k in enumerate(ks):
        for irep in range(nrep):
            score_SC[irep, iz] = results[iz][irep]["SC"][score]
            for ix in range(nx):
                score_values[ix, irep, iz] = results[iz][irep][ix][score]

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    perc = 20
    transparency = 0.2
    cm = colormaps.get_cmap("jet")
    styles = ['-', '--', ':', '-.']

    if relative:
        # plot timings    
        for iz, k in enumerate(ks):
            label = "k = {}".format(k)
            y_values = np.concatenate((score_SC[:,iz].reshape((1,-1)), score_values[:,:,iz]), axis=0)
            plot_perf(rhos, score_values[:,:,iz]/np.mean(score_SC[:,iz]), 'k', label, marker="", style = styles[iz], perc=perc, alpha=transparency, ax=ax, direction=-1)
            plot_perf(rhos, score_values[:,:,iz]/np.mean(score_SC[:,iz]), 'k', "", style="", perc=perc, alpha=transparency, ax=ax, direction=-1)
            ax.set_xticks(rhos)
    else:
        #plot lines
        for iz, k in enumerate(ks):
            label = "k = {}".format(k)
            xvalues = [1]+list(rhos)
            y_values = np.concatenate((score_SC[:,iz].reshape((1,-1)), score_values[:,:,iz]), axis=0)
            plot_perf(xvalues, y_values, 'k', label, marker="", style = styles[iz], perc=perc, alpha=transparency, ax=ax, direction=-1)
        #plot markers
        for iz, k in enumerate(ks):
            if iz==0:
                labelSC = "SC"
                label = "pasco"
            else:
                labelSC = ""
                label = ""
            plot_perf([1], [score_SC[:, iz]], 'r', labelSC, marker="D", errorbar=False, style='', ax=ax, direction=-1)
            plot_perf(rhos, score_values[:,:,iz], 'k', label, errorbar=False, style='', ax=ax, direction=-1)
            ax.set_xticks(xvalues)
    ax.grid()
    if relative:
        ax.set_ylabel("relative "+score)
    else:
        ax.set_ylabel(score)
    ax.set_xlabel(r"$\rho$")
    if "time" in score:
        ax.set_yscale('log')
    else:
        ax.set_ylim(-0.05,1.05)
    plt.tight_layout()
    legend = ax.legend(loc='right', bbox_to_anchor=(1, 0.5), bbox_transform=fig.transFigure, shadow=True)
    plt.subplots_adjust(right=0.77)  # Increase right margin

    figname = ("timings_"+score+"_n"+str(n)+"_ks"+str(ks)+".pdf").replace(" ", "")
    fig.savefig(plots_dir+figname)