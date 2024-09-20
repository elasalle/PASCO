# PASCO
To install pip install -e .

# Libraries 

- `networkx`
- `numpy`
- `scipy`
- `sklearn` 
- `torch`
- `torch_cluster` (only for Graclus)
- `sknetwork` (only for Louvain)
- `graphtools`


In a terminal with `conda` run the following commands : 
- (personal note : on the CBP, we might need `module load conda3/23.3.1` to get `conda`)
- `conda create -n pasco -c conda-forge graph-tool python=3.11`
- `conda activate pasco`
- `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu` (see [documentation](https://pytorch.org/get-started/previous-versions/#v210) to adapt the command to your installation)
- `pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`
- `pip install matplotlib networkx scikit-learn POT pygsp scikit-network infomap leidenalg`
- `pip install -e .`


Without graph-tool you can do : 
- `conda create -n pasco python=3.10`
- `conda activate pasco`
- `pip install numpy==1.26.4 matplotlib networkx scikit-learn POT pygsp scikit-network infomap leidenalg`
- `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu` (see [documentation](https://pytorch.org/get-started/previous-versions/#v210) to adapt the command to your installation)
- `pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`
- `pip install -e .`

With graph-tool and dask :
- `conda create -n pasco -c conda-forge graph-tool python=3.10`
- `conda activate pasco`
- `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu` (see [documentation](https://pytorch.org/get-started/previous-versions/#v210) to adapt the command to your installation)
- `pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`
- `pip install numpy==1.26.4 matplotlib networkx scikit-learn POT pygsp scikit-network infomap leidenalg`
- go to the `Parallel_Structured_Coarse_Grained_Spectral_Clustering` folder and run `pip install -e .`
- `conda install dask`