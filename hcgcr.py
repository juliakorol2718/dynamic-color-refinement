import numpy as np
import networkx as nx
from sympy import prime as pi
import pandas as pd

# color refinement algorithm using hashing
# from the paper "Power Iterated CR"
PRECISION = 7

def log_pi(x):
    #return np.round(np.log(float(pi(int(x)))), PRECISION)
    return np.round(np.log(pi((x))), PRECISION)

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
            Ac = A @ c.apply(log_pi)
        except Exception as e:
            print("Error occurred:", e)
            print(c)
            print(type(c))
        c_hash = (c + Ac).apply(lambda x: np.round(x, PRECISION))
        c, m = colors(c_hash)
        m_pre = m_aft
        m_aft = m
    return c