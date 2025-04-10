# scp elasalle@r740gpu1.cbp.ens-lyon.fr:/projects/users/elasalle/PASCO/experiments/expe_param/results/influence_of_method_align.pickle C:\Users\user\Documents\GitHub\PASCO\experiments\expe_param\results\

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pickle
import argparse

fs = 22

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
    "font.size": fs
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
    score_names = {"ami":"AMI",
                   "ari": "ARI",
                   "time":"time",
                   "modularity":"modularity"}

    # set directories
    res_dir = "results/"
    saving_file_name = res_dir + 'influence_of_' + varying_param + '.pickle'
    plots_dir = "../data/plots/param/"

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

    fig, ax = plt.subplots(1,1, figsize=(7,4), num=varying_param)
    perc = 20
    transparency = 0.2
    cm = colormaps.get_cmap("jet")
    labels_param = {
        "rho" : r"$\rho = $", 
        "n_tables" :  r"$R = $",
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
    ax.set_ylabel(score_names[score])
    ax.set_xlabel(r"$\alpha$", fontsize=fs+4)
    if score=="time":
        ax.set_yscale('log')
    else:
        ax.set_ylim(-0.05,1.05)

    if varying_param=="method_align":
        right = 0.57
    else:
        right = 0.7
    plt.subplots_adjust(left=0.15, bottom=0.2, right=right, top=0.99)  # Increase right margin
    plot_bottom = ax.get_position().y0
    plot_height = ax.get_position().height
    handles, labels = ax.get_legend_handles_labels()
    legend_ax = fig.add_axes([right, plot_bottom, 1-right, plot_height])
    legend_ax.axis('off')  # Hide the axis for the legend area
    legend_ax.legend(handles, labels, loc='center', shadow=True)

    figname = "influence_"+varying_param+"_"+score+"_n"+str(n)+"_k"+str(k)+"_d"+str(d)+".pdf"
    fig.savefig(plots_dir+figname)


    