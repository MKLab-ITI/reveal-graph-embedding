__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import errno


def make_sure_path_exists(path):
    """
    Checks if a directory path exists, otherwise it makes it.

    Input: - path: A string containing a directory path.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
