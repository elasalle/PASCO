from pasco.data_generation import generate_or_import_graph
from pasco.coarsening import Coarsening
from utils import compute_projector, spectral_error
from libraries.coarsening_utils import coarsen
from libraries import graph_lib
import numpy as np
import networkx as nx
from pygsp.graphs import Graph
from scipy.sparse import csr_array, eye
from scipy.linalg import cholesky, svd, eigh
from scipy.sparse.linalg import lobpcg, eigsh
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "font.serif": ["Palatino"],
    "font.size": 12,
    'axes.titlesize': 15,
    'figure.titlesize': 20,
})

if __name__ == '__main__':

    # hyper-parameters
    SBM_graph = True
    rs = np.linspace(0.1,0.9,17)
    n_tables = 10  # number of coarsened graphs
    method_sampling = 'uniform_node_sampling'
    parallel = True
    verbose = False
    save_fig = True

    if SBM_graph:
        # (n,k,d,alpha)
        graphs = [(1000,10,10,1/10), (1000,10,10,1/20), (1000,100,10,1/100), (1000,100,10,1/200)]
    else:
        graphs  = ['yeast','minnesota', 'airfoil'] 
    coarsening_methods = ["pasco", 'variation_edges', 'heavy_edge']
    # error_types = ["RIP", "pseudoRIP", "RSA"]
    error_types = ["RSA"]

    # Data generation
    As, Ls = [], []
    if SBM_graph:
        for n,k,d,alpha in graphs:
            G = generate_or_import_graph(n, k, d, alpha, "../data/", seed=2024)
            A = nx.adjacency_matrix(G, nodelist=range(n))
            As.append(A)
            L = nx.laplacian_matrix(G, nodelist=range(n)) # combinatorial laplacian
            Ls.append(L)
        del G
    else:
        k = 10
        for graph in graphs:
            G = graph_lib.real(10000, graph)
            As.append(csr_array(G.W))
            Ls.append(G.L)
    
    errors = {}
    for A,L,graph in zip(As, Ls, graphs):
        print("Graph : {}".format(graph))

        n = A.shape[0]
        errors[graph] = {}


        # compute the square root of the laplacian
        lambdas, U = eigh(L.toarray())
        lambdas[0] = 0
        S = U @ np.diag(np.sqrt(lambdas)) @ U.T
        # S = cholesky(L.todense())
        print("S computed")
        print("S.T @ S == L :  {}".format(np.linalg.norm(L - S.T @ S) < 1e-5 ))

        # compute the k first (without the first) eigen elements (used for the error computations)
        lambdas_k, Uk = eigsh(csr_array.asfptype(L), k=k, which='SM', tol=1e-6)
        Lambdask_halfinv = np.diag(lambdas_k**(-1/2))
        Lambdask_halfinv = Lambdask_halfinv[1:, 1:] # remove the first element
        Uk = Uk[:,1:] # remove the first element
        # Lambdask_halfinv[0] = 0 # enforce the first eigenvalue to be 0
        print("eigen elements computed")

        
        for cm in coarsening_methods:
            print("coarsening method: {}".format(cm))
            errors[graph][cm] = {}
            for err_type in error_types:
                errors[graph][cm][err_type] = []
            for r in rs:
                ns = int(np.floor((1-r)*n))
                print("  r = {:.0%}".format(r))

                if cm=="pasco":
                    coarsening = Coarsening(ns, 1)
                    Aprimes, tables = coarsening.fit_transform_multiple(A, n_tables)
                    Pis = [compute_projector(table, n, ns) for table in tables]
                else:
                    pygspG = Graph(A)
                    C, _, _, _ = coarsen(pygspG, K=k, r=r, max_levels=20, method=cm, algorithm='greedy')
                    Pis = [C.T @ C]

                for err_type in error_types:
                    err = [ spectral_error(S, Pi, Uk, Lambdask_halfinv, method=err_type) for Pi in Pis]
                    errors[graph][cm][err_type].append(err)

    # figure

    plot_dir = "../data/plots/RSA"
    if SBM_graph:
        fig_name = "RSA_SBM"
    else:
        fig_name = "RSA_Loukas"

    perc = 0.25
    alpha = 0.2
    tab20 = plt.cm.get_cmap('tab10')
    lsty = ['-', '--', ':']
    mkr = ['o', '^', 'P']
    fig, axs = plt.subplots(1,len(graphs), figsize=(5*len(graphs), 7), sharey=True)
    if len(graphs)==1:
        axs = [axs]
    for iax, ax in enumerate(axs):
        graph = graphs[iax]
        for i, cm in enumerate(coarsening_methods):
            for j, err_type in enumerate(error_types):
                errs = errors[graph][cm][err_type]
                ax.plot(rs, np.mean(errs, axis=1), color=tab20(i), marker=mkr[j], linestyle=lsty[j], label="{} ({})".format(err_type, cm))
                ax.fill_between(rs, np.percentile(errs, perc, axis=1), np.percentile(errs, 100 - perc, axis=1), alpha=alpha, facecolor=tab20(i))
        ax.set_yscale('log')
        ax.set_xlabel(r"compression rate $(1-1/\rho)$")
        ax.set_ylabel("spectral error")
        if SBM_graph:
            ax.set_title("(n,k,d,alpha) = {}".format(graph))
        else:
            ax.set_title(graph)
        ax.grid()
        if iax==0:
            legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12), bbox_transform=fig.transFigure, shadow=True, ncol=len(coarsening_methods))
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
    fig.savefig(plot_dir+"/"+fig_name+".pdf")
    plt.show()





    




