from dataclasses import dataclass
import numpy as np

class IterationExperiment:
    '''
    list of iterations and in each of them:
    coloring: pd.DataFrame with columns "color", "hash"
    hash_dict: dict hash -> color, orbit
    nr_of_colors: m
    sym_vertices: set of symmetric vertices
    '''
    def __init__(self, coloring, hash_dict):
        self.coloring = coloring
        self.hash_dict = hash_dict
        self.nr_of_colors = len(hash_dict)

@dataclass
class Iteration:
    color: np.ndarray
    hash: np.ndarray | None = None
    hash_dict: dict | None = None