# PASCO

This repository contains the code to run PASCO (PArallel Structured COarsening): an overlay to speed up graph clustering algorithms. It is associated to the paper: soon to come.

# Installation

To run PASCO, follow the instruction below. \
In a terminal, go into the `PASCO` folder, and run the following commands : 
- `conda create -n pasco -c conda-forge graph-tool`
- `conda activate pasco`
- `pip install matplotlib networkx scikit-learn POT pygsp scikit-network infomap leidenalg pandas`
- `pip install -e .`

The libraries `graph-tool` and `infomap` are sometimes problematic to install. 
If you do not intend to use neither MDL nor infomap as clustering algorithm, you can set a simpler conda environment.
To do so, run the following command from the `PASCO` folder.
- `conda create -n pasco python`
- `conda activate pasco`
- `pip install matplotlib networkx scikit-learn POT pygsp scikit-network leidenalg pandas`
- `pip install -e .`

# Tutorial 

Once the conda environment is installed, you can check our tutorial notebook `tutorial.ipynb` to see how to simply use PASCO (with clustering algorithm already implemented, or your own clustering algorithm). 