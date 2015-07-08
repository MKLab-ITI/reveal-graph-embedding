__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import h5py
import numpy as np
import os


def write_predictions(path, data):
    with h5py.File(path + ".h5", "w") as f:
        f.create_dataset('data', data=data, compression="gzip", compression_opts=9)


def read_predictions(path):
    with h5py.File(path + ".h5") as f:
        data = f["/data"][:, :]

        return data


def npy_to_h5(path):
    data = np.load(path + ".npy")

    write_predictions(path, data)

    os.remove(path + ".npy")
