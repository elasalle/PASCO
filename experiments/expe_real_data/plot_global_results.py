# in anaconda prompt
# scp elasalle@s92gpu2.cbp.ens-lyon.fr:/projects/users/elasalle/Parallel_Structured_Coarse_Grained_Spectral_Clustering/expes/expes_realdata/result_real_data/res_mag_CSC_MDL_SC_infomap_leiden_louvain_ot.pickle C:\Users\user\Documents\GitHub\Parallel_Structured_Coarse_Grained_Spectral_Clustering\expes\expes_realdata\result_real_data

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
    "font.size": 18,
    # 'axes.titlesize': 16,
    # 'figure.titlesize': 20,
})

def create_colormap_to_white(base_color):
    """
    Create a colormap that transitions from the specified base color to white.
    """
    return LinearSegmentedColormap.from_list('custom_colormap', [base_color, 'white'])

if __name__ == '__main__':

    dataset = "mag"
    suffix = "ot"
    solvers = ['SC','CSC','louvain','leiden','MDL','infomap']
    res_dir = "results/"
    fig_dir = "../data/plots/real_data/"
    save = True
    cvxhull = True    
    nt_max = 15
    extra_vmax_coef = 1.05

    sorted_solvers = solvers.copy()
    sorted_solvers.sort()
    saving_file_name = res_dir + '/res_' + dataset + '_' + '_'.join(sorted_solvers) + "_" + suffix +'.pickle' # when using pickle to save the results
    with open(saving_file_name, 'rb') as f:
        results = pickle.load(f)
    # results = np.load(saving_file_name, allow_pickle='TRUE').item()

    # gather parameters
    rhos = list(results[solvers[0]].keys())[1:] # keep only the rhos > 1
    nts = list(results[solvers[0]][rhos[-1]].keys())
    perfs = ["ami", "modularity", "gnCut", "dl"]
    markers = {"leiden" : ">",
               "louvain": "^",
               "SC" : "P",
               "CSC"    : 'X',
               "MDL"    : '*',
               "infomap": 'h',
               "pasco"  : 'o'}
    colors = { "SC" : "green",
               "CSC"    : "lime",
               "louvain": "red",
               "leiden" : "orange",
               "MDL"    : 'blue',
               "infomap": 'darkturquoise'}
    print('rho in ', rhos)
    print('nts in ', nts)
    
    # norm = Normalize(vmin=1, vmax=nt_max*extra_vmax_coef)
    norm = Normalize(vmin=0, vmax=len(nts)) # put the nts uniformly on the colormap scale
    color_values = norm(range(len(nts)))

    
    # fig = plt.figure(figsize=(4*len(perfs), 5*len(rhos)), num='time')
    # gs = gridspec.GridSpec(len(rhos)+1, len(perfs)+1, 
    #                        height_ratios=[1]*len(rhos) + [0.2], width_ratios=[1]*len(perfs) + [0.1], 
    #                        hspace=0.3, wspace=0.45, 
    #                        top=0.85, bottom=0.01, left=0.055, right=0.96)
    # axs = [fig.add_subplot(gs[0, j]) for j in range(len(perfs))]
    # rho = rhos[0] # to change
    # for jax , (ax, perf) in enumerate(zip(axs, perfs)):
    #     for solver in solvers:
    #         nts = list(results[solver][rho].keys())
    #         times = [results[solver][rho][nt]["time"] for nt in nts]
    #         the_perfs = [results[solver][rho][nt][perf] for nt in nts]
    #         solver_time = results[solver][1][1]["time"]
    #         solver_perf = results[solver][1][1][perf]
            
    #         current_cmap = create_colormap_to_white(colors[solver])
    #         c = [current_cmap(val) for val in color_values]

    #         ax.scatter(times, the_perfs, c=c, marker=markers["pasco"])
    #         ax.scatter(solver_time, solver_perf, s=120, marker=markers[solver], color=colors[solver], label=solver)
    #         if cvxhull:
    #             all_times = times + [solver_time]
    #             all_perfs = the_perfs + [solver_perf]
    #             points = np.transpose([all_times, all_perfs])
    #             hull = ConvexHull(points)
    #             ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=colors[solver], alpha=0.15)
    #     if perf in ["modularity", "gnCut", "dl"]:
    #         true_perf = "true_"+perf
    #         ax.axhline(y=results[true_perf], linestyle='--', color="gray")
    #     ax.grid(alpha=0.5)
    #     ax.set_xscale("log")
    #     ax.set_xlabel("time")
    #     ax.set_ylabel(perf)
    #     # ax.set_title("rho = {}".format(rhos[0]))  # should be in the above if, further left than the ylabel

    # # Add the legend
    # handles, labels = axs[0].get_legend_handles_labels()
    # legend_ax = fig.add_subplot(gs[1, :])
    # legend_ax.axis('off')  # Hide the axis for the legend area
    # legend_ax.legend(handles, labels, loc='center', ncol=len(solvers), shadow=True)

    # # Add the colorbar
    # cbar_ax = fig.add_subplot(gs[:len(rhos), len(perfs)])
    # cmap = mpl.colormaps.get_cmap("gray")
    # sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', fraction=0.05, pad=0.05, label=r'$t$')
    # cbar.ax.yaxis.set_label_position('left')
    # cbar.set_ticks(range(len(nts)))
    # cbar.set_ticklabels([str(val) for val in nts])


    # Define the figure and grid layout
    fig = plt.figure(figsize=(4*len(perfs), 5*len(rhos)), num='time')

    # Define the grid for the plots only (without colorbars)
    wspace = 0.5
    bottom = 0.27
    gs = gridspec.GridSpec(len(rhos), len(perfs), 
                        hspace=0.3, wspace=wspace,  # Keep spacing between actual plots
                        top=0.87, bottom=bottom, left=0.055, right=0.87)  # Right boundary at 0.85 to leave space for colorbars

    # Create axes for the main plots
    axs = [fig.add_subplot(gs[0, j]) for j in range(len(perfs))]
    rho = rhos[0]  # Change rho as needed

    # Iterate over performance metrics and solvers
    for jax, (ax, perf) in enumerate(zip(axs, perfs)):
        for solver in solvers:
            nts = list(results[solver][rho].keys())
            times = [results[solver][rho][nt]["time"] for nt in nts]
            the_perfs = [results[solver][rho][nt][perf] for nt in nts]
            solver_time = results[solver][1][1]["time"]
            solver_perf = results[solver][1][1][perf]
            
            current_cmap = create_colormap_to_white(colors[solver])
            c = [current_cmap(val) for val in color_values]
            
            ax.scatter(times, the_perfs, c=c, marker=markers["pasco"])
            ax.scatter(solver_time, solver_perf, s=120, marker=markers[solver], color=colors[solver], label=solver)
            
            if cvxhull:
                all_times = times + [solver_time]
                all_perfs = the_perfs + [solver_perf]
                points = np.transpose([all_times, all_perfs])
                hull = ConvexHull(points)
                ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=colors[solver], alpha=0.15)
        
        if perf in ["modularity", "gnCut", "dl"]:
            true_perf = "true_" + perf
            ax.axhline(y=results[true_perf], linestyle='--', color="gray")
        ax.grid(alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("time")
        if perf in ["ami", "modularity"]:
            ax.set_ylabel(perf+r" $\rightarrow$")
        else:            
            ax.set_ylabel(r"$\leftarrow$ " + perf)

    # Define position for colorbars outside the GridSpec
    colorbar_width = 0.008
    colorbar_bottom = axs[0].get_position().y0
    colorbar_height = axs[0].get_position().height
    rightmost_plot_right = axs[-1].get_position().x1
    colorbar_mostleft = rightmost_plot_right + 0.05

    # Add multiple colorbars (one for each solver)
    for idx, solver in enumerate(solvers):
        # Create the colormap for the solver
        current_cmap = create_colormap_to_white(colors[solver])

        # Calculate the position of the colorbar (outside the grid of the plots)
        colorbar_left = colorbar_mostleft + idx * colorbar_width
        
        # Add a colorbar for each solver
        cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])  # Custom axes for colorbars
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=current_cmap)
        sm.set_array([])

        # Customize the colorbar for each solver
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', fraction=0.05, pad=0.0)

        # Customize ticks and labels based on the position of the colorbar
        if idx == 0:
            # Leftmost colorbar: Add label 't' and place it to the left
            cbar.ax.set_ylabel(r'$t$', rotation=90)
            cbar.ax.yaxis.set_label_position('left')

        # Hide ticks and labels for all colorbars except the rightmost one
        cbar.ax.set_yticklabels([])
        
        if idx != len(solvers) - 1:
            # Remove the ticks for all except the rightmost one
            cbar.ax.tick_params(axis='y', which='both', length=0)  # Hide ticks

    # Only the rightmost colorbar should have ticks and labels
    rightmost_solver = solvers[-1]
    rightmost_cmap = create_colormap_to_white(colors[rightmost_solver])
    rightmost_cbar_ax = fig.add_axes([colorbar_mostleft + (len(solvers) - 1) * colorbar_width , colorbar_bottom, colorbar_width, colorbar_height])
    rightmost_sm = mpl.cm.ScalarMappable(norm=norm, cmap=rightmost_cmap)
    rightmost_sm.set_array([])
    rightmost_cbar = fig.colorbar(rightmost_sm, cax=rightmost_cbar_ax, orientation='vertical', fraction=0.05, pad=0.0)
    rightmost_cbar.set_ticks(range(len(nts)))
    rightmost_cbar.set_ticklabels([str(val) for val in nts])  # Add the ticks only to the rightmost colorbar

    # Add the legend
    handles, labels = axs[0].get_legend_handles_labels()
    # legend_ax = fig.add_subplot(gs[len(rhos), :])
    legend_ax = fig.add_axes([0, 0.02, 1, 0.1])
    legend_ax.axis('off')  # Hide the axis for the legend area
    legend_ax.legend(handles, labels, loc='center', ncol=len(solvers), shadow=True)






    plt.suptitle("Graph : {}".format(dataset))

    if save:
        fig.savefig(fig_dir + "simple_perf_vs_"+ "time" + "_" + dataset+'_' + '_'.join(solvers)+ "_" + suffix +".pdf")

    # plt.show()
