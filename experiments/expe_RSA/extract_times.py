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

    index = 13
    print("r = {} \n".format(rs[index]))

    print('& ' + ' & '.join(coarsening_methods) + r" \\ \hline")
    for graph_type in graph_types:
        for graph in timings[graph_type].keys():
            if isinstance(graph, str):
                graph_name = graph
            else:
                graph_name = "SSBM"+str((graph[1], graph[3]))
            toprint = graph_name + " & "
            for cm in coarsening_methods:
                times = timings[graph_type][graph][cm][index]
                mean = np.format_float_scientific(np.mean(times), precision=1, exp_digits=1)
                std = np.format_float_scientific(np.std(times), precision=1, exp_digits=1)
                toprint += mean + " (" + std +") & "
            print(toprint[:-2] + r" \\ \hline")
              

    