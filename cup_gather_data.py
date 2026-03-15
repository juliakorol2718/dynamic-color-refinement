import numpy as np
import networkx as nx
import pandas as pd
from sympy import prime as pi

LOG_PI_TABLE = np.load("log_pi_table_big.npy") #np.load("log_pi_table.npy")
PRECISION = 7
'''
Iteration class:
list of iterations and in each of them:
coloring: pd.DataFrame with columns "color", "hash"
hash_dict: dict hash -> color, orbit
nr_of_colors: m
sym_vertices: set of symmetric vertices
'''

class Iteration:
    def __init__(self, coloring, hash_dict):
        self.coloring = coloring
        self.hash_dict = hash_dict
        self.nr_of_colors = len(hash_dict)

# c - pd.Series with colors (numbers), vertex -> color

PRECISION = 7
def log_pi(x):
    return np.round(np.log(float(pi(int(x)))), PRECISION)

def factorize_hashes(c_hash):
    c, uniques = pd.factorize(c_hash)
    return pd.Series(c + 1), uniques

def build_hash_dict(c_hash, uniques):
    hash_dict = {}
    
    for color, hash_val in enumerate(uniques):
        mask = c_hash == hash_val
        orbit = set(c_hash.index[mask])
        hash_dict[str(hash_val)] = {
            'color': color + 1,
            'orbit': orbit if orbit else {None}
        }
    
    return hash_dict

def colors_data(c_hash):
    c, uniques = factorize_hashes(c_hash)
    hash_dict = build_hash_dict(c_hash, uniques)
    return c, hash_dict


def hcgcr_data(G):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0, n)])
    c = np.ones(n, dtype=int)
    iterations = []
    
    m_pre, m_aft = -1,1
    while m_pre != m_aft:
        c_log_pi = LOG_PI_TABLE[c]
        Ac = A @ c_log_pi
        c_hash = pd.Series(c + Ac)
        c, hash_dict = colors_data(c_hash)
        
        coloring = {'color': c, 'hash': c_hash}
        coloring = pd.DataFrame(coloring)

        m_pre = m_aft
        m_aft = len(hash_dict)

        if m_pre != m_aft:
            iterations.append(Iteration(coloring, hash_dict))
            c = c.to_numpy(dtype=int)
    
    return iterations