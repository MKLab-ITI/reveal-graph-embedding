# reveal-graph-embedding

Implementation of community-based graph embedding for user classification.

Features
--------
- Graph-based, multi-label user classification.
- Implementation of the ARCTE (Absorbing Regularized Commute Times Embedding) algorithm for graph-based feature extraction.
- Both python vanilla and cython-optimized versions.
- Implementation of other feature extraction methods for graphs (Laplacian Eigenmaps, Louvain, MROC).
- Evaluation score and time benchmarks.

Install
-------
### Required packages
- numpy
- scipy
- h5py
- scikit-learn
- Cython
- networkx
- python-louvain

### Installation
To install for all users on Unix/Linux:

    python3.4 setup.py build
    sudo python3.4 setup.py install
  
Alternatively:

    pip install reveal-graph-embedding
