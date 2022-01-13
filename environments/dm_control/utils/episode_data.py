"""Container for utils to store episode data for debugging"""
import numpy as np

def pack_datadict(datadict):
    """
    Pack data dictionary into numpy array.
    Assume all value in dictionary have the same length time lenght.
    """
    return {key: np.asarray([ts[key] for ts in datadict])
            for key in datadict[0]}
