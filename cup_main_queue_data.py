import numpy as np
import networkx as nx
from cup_classes import Iteration
from cup_functions import update_hash_value, update_colors, continue_color_refinement
from typing import Callable


def color_update_propagation(G_upd: nx.Graph, S: set, old_iterations: list, queue_update_function: Callable) -> tuple:
    """Update the color refinement coloring of a graph after a graph modification.
    Additionally, keep track of the size of the queue in each iteration.

    Args:
        G_upd (nx.Graph): The updated graph.
        S (set): Set of vertices whose incident edges were inserted or removed.
        old_iterations (list):  List of Iteration objects representing the color 
                                refinement process on the original graph.
        queue_update_function (Callable):   Function that determines how the queue 
                                            for the next iteration is updated. 
                                            This allows different update strategies 
                                            (CUP, CAS, COR).

    Returns:
        tuple: A tuple containing a list of coloring vectors and a list of queue 
                lengths representing the updated color refinement process.
    """
    
    q = set(S)
    n = G_upd.number_of_nodes()
    A = nx.adjacency_matrix(G_upd, nodelist = [i for i in range(0, n)]).toarray()
    k = len(old_iterations)
    coloring_upd_list = []
    q_len_list = []
    
    def q_size(q):
        if isinstance(q, set):
            return len(q)
        if isinstance(q, int):
            return 1
        return 0
    
    it_prev = Iteration(color=np.ones(n, dtype=int))
    i = 0
    m_pre, m_aft = -1, 1
    while i < k and m_pre != m_aft and len(q) != 0:
                
        it_old = old_iterations[i]
        q_len_list.append(q_size(q))
        q_list = sorted(list(q))
        
        # update the hashes of vertices in the queue
        hash_upd = update_hash_value(q_list, A, it_old.hash, it_prev)
        # update the colors according to the new hashes
        it_upd = update_colors(q_list, it_old, hash_upd)
        # build the next queue
        q, it_upd = queue_update_function(q_list, G_upd, it_upd, it_old)

        m_pre = m_aft
        m_aft = len(it_upd.hash_dict)
        coloring_upd_list.append(it_upd.color)
        it_prev = it_upd
        i += 1
    
    # if len(q) == 0 it means that the change in the graph did not affect the coloring
    if len(q) == 0:
        for i in range(i,k):
            it_old = old_iterations[i]
            coloring_upd_list.append(it_old.color)
            q_len_list.append(0)

    # if m_pre != m_after it means that the coloring is not finished
    elif m_pre != m_aft:
        next_colorings = continue_color_refinement(G_upd, it_upd.color, m_pre, m_aft)
        coloring_upd_list.extend(next_colorings)
        q_len_list.extend([n]*(len(next_colorings)))
        
    return coloring_upd_list, q_len_list