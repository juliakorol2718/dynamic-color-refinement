import numpy as np
import networkx as nx
import pandas as pd
from dataclasses import dataclass
from hcgcr import colors

LOG_PI_TABLE = np.load("log_pi_table.npy")
PRECISION = 7

@dataclass
class ColoringState:
    color: np.array
    hash: np.array
    hash_dict: dict  

def update_hash_value(q, A, hash_old, c_pre):
    """_summary_

    Args:
        q (list): _description_
        A (np matrix): adjacency
        hash_old (np array): columns with string hashes
        c_pre (_type_): _description_

    Returns:
        np.array float: updated hashes
    """
    if not q:
        return hash_old.copy()

    color_pre = c_pre["color"].to_numpy(dtype=np.int64)
    c_log_pi = LOG_PI_TABLE[color_pre]

    trans = A[q, :] @ c_log_pi
    trans = np.asarray(trans).ravel()

    new_hash = np.round(color_pre[q] + trans, PRECISION)

    hash_up = hash_old.copy()
    hash_up[q] = new_hash.astype(str)
    return hash_up


def update_colors(q, state_old, hash_up):
    """

    Args:
        q (list)
        coloring_old (_type_): _description_
        coloring_up (_type_): _description_
        hash_dict_old (bool): _description_

    Returns:
        _type_: _description_
    """
    hash_old = state_old.hash
    hash_dict_old = state_old.hash_dict
    color_old = state_old.color
    color_up = color_old.copy()
    
    if not q:
        return ColoringState(color=color_up, hash_dict=hash_dict_old, hash=hash_old)

    h_old_q = hash_old[q]
    h_up_q  = hash_up[q]
    
    hash_dict_up = {
        h: {
            "color": data["color"], 
            "orbit": set(data["orbit"])}
        for h, data in hash_dict_old.items()
    }
    free_color = len(hash_dict_up) + 1
    
    # remove old hashes
    for v, h_old in zip(q, h_old_q):
        orbit_old = hash_dict_up[h_old]['orbit']
        orbit_old.discard(v)
        
        if len(orbit_old) == 0:
            del hash_dict_up[h_old]
    
    # add new hashes
    for v, h_up in zip(q, h_up_q):
        if h_up in hash_dict_up:
            hash_dict_up[h_up]['orbit'].add(v)
        else:
            hash_dict_up[h_up] = {
                'color': free_color,
                'orbit': {v}
            }
            free_color += 1
            
        color_up[v] = hash_dict_up[h_up]['color']
        
    return ColoringState(color=color_up, hash_dict=hash_dict_up, hash=hash_up)
########################################################################################################################################################################
# UPDATE FUNCTIONS
########################################################################################################################################################################


# Update queue - basic version

def update_queue_add_all_neighbors(q_list, G, state_up, state_old):
    adj = G.adj
    q_new = set(q_list)
    for v in q_list:
        q_new.add(v)
        q_new.update(adj[v])
    return q_new, state_up
########################################################################################################################################################################


# Update queue - remove asymmetric vertices
def update_queue_remove_asymmetric(q_list, G, state_up, state_old):
    q_new = set()
    asymmetric_vertices = set()
    adj = G.adj

    h_old_q = state_old.hash[q_list].astype(str)
    h_up_q  = state_up.hash[q_list].astype(str)

    adj = G.adj

    for v, h_old, h_up in zip(q_list, h_old_q, h_up_q):
        orbit_old = state_old.hash_dict[h_old]["orbit"]
        orbit_up  = state_up.hash_dict[h_up]["orbit"]
        if len(orbit_old) == 1 and len(orbit_up) == 1:
            asymmetric_vertices.add(v)
        else:
            q_new.add(v)
            q_new.update(adj[v])
    q_new.difference_update(asymmetric_vertices)

    recolor_unchanged_orbits(asymmetric_vertices, state_up, state_old)
    
    return q_new, state_up



########################################################################################################################################################################
# Update queue - remove orbits

def update_queue_remove_unchanged_orbits(q_list, G, state_up, state_old):
    
    unchanged_orbits, asymmetric_vertices = get_unchanged_orbits(q_list, state_up, state_old)
    recolor_unchanged_orbits(unchanged_orbits, state_up, state_old)
    
    q_new = set()
    adj = G.adj
    affected_vertices = set(q_list).difference(unchanged_orbits)
    for v in affected_vertices:
        q_new.add(v)
        q_new.update(adj[v])
        
    q_new.difference_update(asymmetric_vertices)
    
    return q_new, state_up

def get_unchanged_orbits(q, state_up, state_old):
    q_set = set(q)
    unchanged_orbits = set()
    seen_vertices = set()
    asymmetric_vertices = set()

    for v in q:
        if v in seen_vertices:
            continue
        
        h_old = state_old.hash[v]
        h_up  = state_up.hash[v]

        orbit_old = state_old.hash_dict[h_old]['orbit']
        orbit_up  = state_up.hash_dict[h_up]['orbit']
        
        # Mark both sides as processed so we don't revisit the same cells
        seen_vertices.update(orbit_old)
        seen_vertices.update(orbit_up)

        if orbit_up == orbit_old and orbit_old.issubset(q_set):
            unchanged_orbits.update(orbit_old)
            if len(orbit_up) == 1:
                asymmetric_vertices.add(v)

    return unchanged_orbits, asymmetric_vertices

def recolor_unchanged_orbits(unchanged_orbits, state_up, state_old):
    if not unchanged_orbits:
        return
    
    unchanged = list(unchanged_orbits)
    
    hash_dict_up = state_up.hash_dict
    
    for v in unchanged:
        h_old = state_old.hash[v]
        h_up = state_up.hash[v]

        # Update hash_dict if this hash hasn't been processed
        if h_up in hash_dict_up:
            orbit = hash_dict_up[h_up]['orbit']
            del hash_dict_up[h_up]

            if h_old not in hash_dict_up:
                hash_dict_up[h_old] = {
                    'color': state_old.color[v],
                    'orbit': set()
                }
            hash_dict_up[h_old]['orbit'].update(orbit)
    
    # revert color and hash:
    state_up.color[unchanged] = state_old.color[unchanged]
    state_up.hash[unchanged] = state_old.hash[unchanged]

def continue_color_refinement(G, c, m_pre, m_aft):
    colorings_up_list = []
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0,n)]) 
    c = np.array(c)
    while m_pre != m_aft:
        c_log_pi = LOG_PI_TABLE[c]
        Ac = A @ c_log_pi
        c, m = colors(c + Ac)
    
        m_pre = m_aft
        m_aft = m
        if m_pre != m_aft:
            colorings_up_list.append(list(c))
            
    return colorings_up_list


if __name__ == "__main__":
    print("this is main")
