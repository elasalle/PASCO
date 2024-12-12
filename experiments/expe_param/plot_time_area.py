import numpy as np
import matplotlib as mpl
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Test the influence of k pasco performances.')
    parser.add_argument('-ik', '--indexk', nargs='?', type=int, default=0)
    args = parser.parse_args()
    iz = args.indexk

    # set directories
    res_dir = "results/"
    saving_file_name = res_dir + 'timings' + '.pickle'
    plots_dir = "../data/plots/param/"

    # load results
    with open(saving_file_name, 'rb') as f:
        results = pickle.load(f)

    n = results["n"]
    k = results["ks"][iz]
    d = np.rint(results["avg_d"]/np.log(n)*2)/2
    rhos = [3,5,10,15] 
    # rhos = results["rhos"] # for the next run

    nx = len(rhos)
    nrep = 10
    # nrep = results["nrep"]

    co_times = np.zeros(nx+1)
    cl_times = np.zeros(nx+1)
    fu_times = np.zeros(nx+1)
    
    cl_times[0] = np.mean([results[iz][irep]["SC"]["time"] for irep in range(nrep)])
    for ix in range(nx):
        co_times[ix+1] = np.mean([results[iz][irep][ix]["coarsening"] for irep in range(nrep)])
        cl_times[ix+1] = np.mean([results[iz][irep][ix]["clustering"] for irep in range(nrep)]) 
        fu_times[ix+1] = np.mean([results[iz][irep][ix]["fusion"] for irep in range(nrep)])

    xvalues = [1]+list(rhos)

    # Compute the cumulative sum
    y1_cumsum = co_times
    y2_cumsum = co_times + cl_times
    y3_cumsum = co_times + cl_times + fu_times

    fig, ax = plt.subplots(1,1, figsize=(7,4))

    # Create the stacked area plot
    ax.fill_between(xvalues, 0, y1_cumsum, label='coarsening', alpha=0.75)
    ax.fill_between(xvalues, y1_cumsum, y2_cumsum, label='clustering', alpha=0.75)
    ax.fill_between(xvalues, y2_cumsum, y3_cumsum, label='fusion', alpha=0.75)

    # Add labels and legend
    ax.set_xlabel(r"$\rho = R$")
    ax.set_ylabel('time')
    ax.set_xticks(xvalues)
    ax.legend()
    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.95, top=0.99)

    # Save plot
    figname = "timings_stacked_area"+"_n"+str(n)+"_k"+str(k)+".pdf"
    plt.savefig(plots_dir+figname)