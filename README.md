# reveal-graph-embedding

Implementation of community-based graph embedding for user classification.

Features
--------
- Graph-based, multi-label user classification experiment demo.
- Implementation of the ARCTE (Absorbing Regularized Commute Times Embedding) algorithm for graph-based feature extraction.
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

User Classification Experiments
-------------------------------
### Get datasets
- SNOW2014Graph dataset: Included anonymized in this project.
- The Arizona State University social computing data repository contains the [ASU-Flickr](http://socialcomputing.asu.edu/datasets/Flickr) and [ASU-YouTube](http://socialcomputing.asu.edu/datasets/YouTube2) datasets.
- The Insight Project Resources repository contains the [Multiview datasets](http://mlg.ucd.ie/aggregation/index.html) in which the PoliticsUK dataset can be found.

### Feature extraction methods
- Implemented methods: ARCTE, BaseComm, LapEig, RepEig, Louvain, MROC.
- Other methods' implementations: [DeepWalk](https://github.com/phanein/deepwalk), [EdgeCluster](http://leitang.net/social dimension.html), [RWModMax](https://github.com/rdevooght/RWModMax), [BigClam](http://snap.stanford.edu/), [OSLOM](http://www.oslom.org/).

### Experiments demo
- Follow instructions on file: reveal_graph_embedding/experiments/demo.py


ARCTE algorithm
---------------
- If you installed the package, you will have an installed script called arcte.
- The source is located in reveal_graph_embedding/entry_points/arcte.py


