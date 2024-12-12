# in anaconda prompt
# scp elasalle@apollo1024g.cbp.ens-lyon.fr:/projects/users/elasalle/PASCO/experiments/expe_real_data/results/res_products_CSC_MDL_SC_infomap_leiden_louvain_ot.pickle C:\Users\user\Documents\GitHub\PASCO\experiments\expe_real_data\results\

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
    "font.size": 26,
    # 'axes.titlesize': 16,
    # 'figure.titlesize': 20,
})

def create_colormap_to_white(base_color):
    """
    Create a colormap that transitions from the specified base color to white.
    """
    return LinearSegmentedColormap.from_list('custom_colormap', [base_color, 'white'])

def log_lower_rounding(x):
    power = np.floor(np.log10(x))
    factor = np.floor(x / 10**power)
    return factor * 10**power

def log_upper_rounding(x):
    power = np.floor(np.log10(x))
    factor = np.ceil(x / 10**power)
    return factor * 10**power


if __name__ == '__main__':

    datasets = ["arxiv", "mag", "products"]
    solvers_to_plot = ['SC','CSC','louvain','leiden','MDL','infomap']
    solvers = ['SC','CSC','louvain','leiden','MDL','infomap']
    suffix = "ot"
    res_dir = "results/"
    fig_dir = "../data/plots/real_data/"
    save = True
    cvxhull = True    
    display_time = "relative"
    display_perf = "diff_rel"
    nt_max = 15
    extra_vmax_coef = 1.05

    sorted_solvers = solvers.copy()
    sorted_solvers.sort()
    results = {}
    for dataset in datasets:
        saving_file_name = res_dir + '/res_' + dataset + '_' + '_'.join(sorted_solvers) + "_" + suffix +'.pickle' # when using pickle to save the results
        with open(saving_file_name, 'rb') as f:
            results[dataset] = pickle.load(f)

    # gather parameters
    rho = 10
    nts = list(results[datasets[0]][solvers[0]][rho].keys())
    perfs = ["ami", "modularity", "gnCut", "dl"]
    perfs_name = {"ami"       : "ami",
                  "modularity": "mod",
                  "gnCut"     : "gnCut",
                  "dl"        : "dl"}
    markers = {"arxiv"    : "P",
               "mag"      : "*",
               "products" : "^",
               "pasco"    : "o"}
    colors = { "arxiv"   : "blue",
               "mag"     : "red",
               "products": "green"}
    

    norm = Normalize(vmin=0, vmax=len(nts)) # put the nts uniformly on the colormap scale
    color_values = norm(range(len(nts)))

    bottom = 0.12
    fig, axss = plt.subplots(
        len(solvers_to_plot), len(perfs), 
        figsize=(18, 22),
        sharey=True, sharex="col")
    plt.subplots_adjust(top= 0.99, bottom=bottom, left=0.11, right=0.99, 
                        hspace=0.12, wspace=0.12)
    for i, (axs,solver) in enumerate(zip(axss,solvers_to_plot)):
        for jax, (ax, perf) in enumerate(zip(axs, perfs)):

            if perf!="ami" and display_perf == "diff_rel":
                ax.axvline(x=0, linestyle='--', linewidth=2, color="gray", zorder=0)
            if display_time == "relative":
                ax.axhline(y=1, linestyle='--', linewidth=2, color="gray", zorder=0)
            ax.grid(alpha=0.5)
            ax.set_yscale("log")

            for ds in datasets:
                nts = list(results[ds][solver][rho].keys())
                solver_perf = results[ds][solver][1][1][perf]
                solver_time = results[ds][solver][1][1]["time"]
                
                if display_time=="relative":
                    times = [results[ds][solver][rho][nt]["time"]/solver_time for nt in nts]
                    solver_time = 1
                else:
                    times = [results[ds][solver][rho][nt]["time"] for nt in nts]
                
                if perf=="ami":
                    the_perfs = [results[ds][solver][rho][nt][perf] for nt in nts]
                else:
                    if display_perf == "abs_diff_rel":
                        true_perf = results[ds]["true_"+perf]
                        the_perfs = [np.abs(results[ds][solver][rho][nt][perf] - true_perf)/true_perf for nt in nts]
                        solver_perf = np.abs(solver_perf - true_perf)/true_perf
                    elif display_perf == "diff_rel":
                        true_perf = results[ds]["true_"+perf]
                        the_perfs = [(results[ds][solver][rho][nt][perf] - true_perf)/true_perf for nt in nts]
                        solver_perf = (solver_perf - true_perf)/true_perf
                    elif display_perf == "diff":
                        true_perf = results[ds]["true_"+perf]
                        the_perfs = [results[ds][solver][rho][nt][perf] - true_perf for nt in nts]
                        solver_perf = solver_perf - true_perf
                    else:
                        the_perfs = [results[ds][solver][rho][nt][perf] for nt in nts]

                current_cmap = create_colormap_to_white(colors[ds])
                c = [current_cmap(val) for val in color_values]
                
                ax.scatter(the_perfs, times, c=c, marker=markers["pasco"])
                ax.scatter(solver_perf, solver_time, s=120, marker=markers[ds], color=colors[ds], label=ds)
                
                if cvxhull:
                    all_times = times + [solver_time]
                    all_perfs = the_perfs + [solver_perf]
                    points = np.transpose([all_perfs, all_times])
                    hull = ConvexHull(points)
                    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=colors[ds], alpha=0.15)
            if i==len(solvers_to_plot)-1:
                if perf=="ami":
                    ax.set_xlabel(perfs_name[perf] + r" $\rightarrow$")
                else:
                    if display_perf == "abs_diff_rel":
                        ax.set_xlabel(r"$\leftarrow$" + " RE " + perfs_name[perf])
                    elif display_perf == "diff_rel":
                        ax.set_xlabel("rel. diff. " + perfs_name[perf])
                    elif display_perf == "diff":
                        ax.set_xlabel("diff " + perfs_name[perf])
                    else:
                        ax.set_xlabel(perf)
            if jax==0:
                if display_time=="relative":
                    ax.set_ylabel(solver + "\n rel. time")
                else:
                    ax.set_ylabel(solver + "\n time")
            


    
    # get left and right sizes:
    mostleft  = axss[-1][0].get_position().x0
    mostright = axss[-1][-1].get_position().x1
    length = mostright - mostleft
    top = bottom-0.03

    # colorbar position
    hmargin = 0.05
    colorbar_left = mostleft + length/2 + hmargin
    colorbar_width = length/2 - 2*hmargin
    colorbar_height = 0.008
    total_height = len(datasets)*colorbar_height
    colorbar_mostbottom = (top - total_height)/2


    #     # Add multiple colorbars (one for each dataset)
    for idx, ds in enumerate(datasets):
        # Create the colormap for the dataset
        current_cmap = create_colormap_to_white(colors[ds])

        # Calculate the position of the colorbar (outside the grid of the plots)
        colorbar_bottom = colorbar_mostbottom + idx*colorbar_height
        
        # Add a colorbar for each dataset
        cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])  # Custom axes for colorbars
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=current_cmap)
        sm.set_array([])

        # Customize the colorbar for each dataset
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', fraction=0.05, pad=0.0)

        # Customize ticks and labels based on the position of the colorbar
        if idx == len(datasets)-1:
            # Leftmost colorbar: Add label 't' and place it to the left
            cbar.ax.set_xlabel(r'$R$')
            cbar.ax.xaxis.set_label_position('top')
    
        if idx != 0:
            # Remove the ticks for all except the rightmost one
            cbar.ax.set_xticklabels([])
            cbar.ax.tick_params(axis='y', which='both', length=0)  # Hide ticks
        else:
            cbar.ax.set_xticks(range(len(nts)))
            cbar.ax.set_xticklabels([str(val) for val in nts])

    # Add the legend
    handles, labels = axss[0][0].get_legend_handles_labels()
    legend_ax = fig.add_axes([mostleft, 0, length/2, top])
    legend_ax.axis('off')  # Hide the axis for the legend area
    legend_ax.legend(handles, labels, loc='center', ncol=len(datasets), shadow=True)

    if save:
        plt.savefig(fig_dir + display_time + "_time_vs_" + display_perf + "_perf_" + '_'.join(datasets) + "_" + suffix +".pdf")
