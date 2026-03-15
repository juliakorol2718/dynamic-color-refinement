import numpy as np
import networkx as nx
from dataclasses import dataclass
import pandas as pd
from cup_gather_data import hcgcr_data
from cup_functions_faster import ColoringState, update_hash_value, update_colors, continue_color_refinement
from typing import Callable


def updateCR(G : nx.Graph, S : set, iterations : list, queue_update_function : Callable) -> list :
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
    iterations = iterations + [iterations[-1]]
    k = len(iterations)
    colorings_up_list = []
    q_len_list = []
    coloring_up_pre = pd.DataFrame([{'color': 1, 'hash': 1} for _ in range(n)])

    def q_size(q):
        if isinstance(q, set):
            return len(q)
        if isinstance(q, int):
            return 1
        return 0
    
    i = 0
    m_pre, m_aft = -1, 1
    while i < k and m_pre != m_aft and len(q) != 0:
                
        q_len_list.append(q_size(q))
        it = iterations[i]
        hash_dict_old = it.hash_dict
        coloring_old = it.coloring
        # convert dfs to np arrays
        q_list = sorted(list(q))
        color_old = coloring_old['color'].to_numpy(dtype=np.int64)
        hash_old = coloring_old['hash'].astype(str).to_numpy()
        state_old = ColoringState(color=color_old, hash=hash_old, hash_dict=hash_dict_old)
        
        # update the hashes of vertices in the queue
        hash_up = update_hash_value(q_list, A, hash_old, coloring_up_pre)
        # update the colors according to hashes
        state_up = update_colors(q_list, state_old, hash_up)
        # build the next queue
        q, state_up = queue_update_function(q_list, G, state_up, state_old)

        # coloring to df
        coloring_up = pd.DataFrame({
            'color': state_up.color, 
            'hash': state_up.hash
            }).astype({'color': 'int64', 'hash': 'string'})
        
        m_pre = m_aft
        m_aft = len(state_up.hash_dict)
        colorings_up_list.append(coloring_up['color'].to_list())
        coloring_up_pre = coloring_up
        i += 1
    
    # if len(q) == 0 it means that change in the graph did not affect the coloring
    if len(q) == 0:
        for i in range(i,k):
            it = iterations[i]
            colorings_up_list.append(it.coloring['color'].to_list())
            q_len_list.append(0)
    # if m_pre != m_after it means that the coloring is not finished
    elif m_pre != m_aft:
        next_colorings = continue_color_refinement(G, coloring_up['color'], m_pre, m_aft)
        colorings_up_list.extend(next_colorings)
        q_len_list.extend([n]*(len(next_colorings)+1))
        
    return colorings_up_list, q_len_list