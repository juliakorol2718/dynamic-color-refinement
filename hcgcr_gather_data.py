import pandas as pd

import numpy as np
import networkx as nx
from cup_classes import Iteration
from pathlib import Path
import numpy as np

# Precomputed values of log(π(i)), where π(i) is the i-th prime number.
# The table allows fast lookup of log(prime_i) for color i without
# recomputing primes or logarithms during refinement.
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PI_TABLE = np.load(PROJECT_ROOT / "log_pi_table.npy")

PRECISION = 7

def factorize_hashes(hash: np.ndarray):
    """Map hashes to integer colors."""
    coloring, uniques = pd.factorize(hash)
    return coloring + 1, uniques

def build_hash_dict(hash: np.ndarray, uniques: np.ndarray):
    """
    Construct a dictionary mapping each hash value to:
        - its assigned integer color
        - the set (orbit) of vertices having that color.
    """
    
    hash_dict = {}
    node_idx = np.arange(len(hash))

    for color, hash_val in enumerate(uniques, start=1):
        orbit = set(node_idx[hash == hash_val])
        hash_dict[hash_val] = {
            "color": color,
            "orbit": orbit if orbit else {None},
        }

    return hash_dict


def build_color_partition(hash: np.ndarray):
    """Compute color classes induced by hash values."""
    c, uniques = factorize_hashes(hash)
    hash_dict = build_hash_dict(hash, uniques)
    return c, hash_dict

def hcgcr_data(G: nx.Graph):
    """
    Compute the color refinement sequence for a graph using the
    power-iterated color refinement (HCGCR) procedure. Algorithm
    taken from an article "Power Iterated Color Refinement" 
    (Kersting et.al., 2014).

    The process starts from a uniform coloring in which all vertices
    get the same color. In each iteration, a hash value is computed
    for every vertex based on its current color and the a colors of 
    its neighbors (see the article for details). Vertices with 
    identical hashes form color classes.

    These classes are stored in a dictionary mapping each hash value to
    its assigned color and the corresponding orbit (the set of vertices
    having that hash). This dictionary represents the partition of the
    vertex set induced by the hashes.

    The process continues until the number of color classes stabilizes.

    Returns
        list[Iteration]
            Sequence of iteration steps containing the coloring, hash
            values, and hash dictionaries at each refinement step.
    """
    
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist=list(range(n)))
    c = np.ones(n, dtype=int)
    iterations = []

    m_pre, m_aft = -1, 1
    while m_pre != m_aft:
        c_log_pi = LOG_PI_TABLE[c]
        Ac = A @ c_log_pi
        hash = np.round(c + Ac, PRECISION)

        c, hash_dict = build_color_partition(hash)

        result_it = Iteration(
            color= c.copy(),
            hash = hash.copy(),
            hash_dict=hash_dict
        )

        m_pre = m_aft
        m_aft = len(hash_dict)
        iterations.append(result_it)

    return iterations