from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd

# color refinement algorithm (HCGCR) using hashing
# from the paper "Power Iterated CR"
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PI_TABLE = np.load(PROJECT_ROOT / "log_pi_table.npy")
PRECISION = 7

def colors(v_hash): # v - vector of hashed values of colors
    c, u = pd.factorize(v_hash)
    m = len(u)
    return pd.Series(c+1), m

def hcgcr(G):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0,n)])
    c = pd.Series(np.ones(n, dtype=int))
    
    m_pre, m_aft = -1, 1
    while m_pre != m_aft:
        
        try:
            c_log_pi = LOG_PI_TABLE[c]
            Ac = A @ c_log_pi
        except Exception as e:
            print("Error occurred:", e)
            print(c)
            print(type(c))
        c_hash = np.round(c + Ac, PRECISION)
        c, m = colors(c_hash)
        m_pre = m_aft
        m_aft = m
    return c