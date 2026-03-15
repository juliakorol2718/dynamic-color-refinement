import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from cup_gather_data import hcgcr_data
from archive.cup_functions import *
from copy import deepcopy
import typing

def updateCR(G : nx.Graph, S : set, iterations : list, queue_update_function : function) -> list :
    """_summary_

    Args:
        G (nx.Graph): _description_
        S (set): _description_
        iterations (_type_): _description_
        queue_update_function (_type_): _description_

    Returns:
        list: _description_
    """
    # G - updated graph
    # S - a set of vertices affected by the changes made in the graph
    # iterations - list of Iteration objects
    
    q = set(S)
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0, n)]).toarray()
    k = len(iterations)
    colorings_up_list = []
    coloring_up_pre = pd.DataFrame([{'color': 1, 'hash': 1} for _ in range(n)])

    i = 0
    m_pre, m_aft = -1, 1
    while i < k and m_pre != m_aft and len(q) != 0:
        it = iterations[i]
        coloring_old = it.coloring
        hash_dict_old = it.hash_dict
        
        coloring_up = update_hash_value(q, A, coloring_old, coloring_up_pre)
        coloring_up, hash_dict_up = update_colors(q, coloring_old, coloring_up, hash_dict_old)
        q, coloring_up, hash_dict_up = queue_update_function(q, G, coloring_old, coloring_up, hash_dict_old, hash_dict_up)

        colorings_up_list.append(coloring_up)
        coloring_up_pre = coloring_up
        m_pre = m_aft
        m_aft = len(hash_dict_up)
        i += 1
    
    # if len(q) == 0 it means that change in the graph did not affect the coloring
    if len(q) == 0:
        for i in range(i,k):
            it = iterations[i]
            colorings_up_list.append(it.coloring)
    # if m_pre != m_after it means that the coloring is not finished
    elif m_pre != m_aft:
        next_colorings = continue_color_refinement(G, coloring_up['color'], m_pre, m_aft)
        colorings_up_list.extend(next_colorings)
        
    return colorings_up_list