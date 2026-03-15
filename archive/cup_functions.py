import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from cup_gather_data import hcgcr_data, log_pi
from hcgcr import colors
from copy import deepcopy


PRECISION = 7



def update_hash_value(q, A, c_old, c_pre):
    c_log_pi = c_pre['color'].apply(log_pi)
    c_up = deepcopy(c_old)
    for v in q:
        v_row_transformed = A[v] @ c_log_pi.values
        new_hash = np.round(c_pre.at[v, 'color'] + v_row_transformed, PRECISION)
        c_up.at[v, 'hash'] = new_hash
    return c_up

# Update hashes and colors
def update_colors(q, coloring_old, coloring_up, hash_dict_old):
    hash_dict_up = deepcopy(hash_dict_old)
    free_color = len(hash_dict_up) + 1
    for v in q:
        h_old, h_up = get_hash_pair(v, coloring_old, coloring_up)
        hash_dict_up = remove_old_hash(v, h_old, hash_dict_up)
        hash_dict_up, free_color = add_new_hash(v, h_up, hash_dict_up, free_color)
        coloring_up.at[v, 'color'] = hash_dict_up[h_up]['color']
        
    return coloring_up, hash_dict_up

# Managing changes of hashes in dictionaries
def get_hash_pair(v, coloring_old, coloring_up):
    return str(coloring_old.at[v, 'hash']), str(coloring_up.at[v, 'hash'])

# Consequences of changing a hash - past
def remove_old_hash(v, h_old, hash_dict_up):
    orbit_old = hash_dict_up[h_old]['orbit']
    orbit_old.discard(v)
    
    if len(orbit_old) == 0:
        del hash_dict_up[h_old]
        
    return hash_dict_up

# Consequences of changing a hash - present
def add_new_hash(v, h_up, hash_dict_up, free_color):
    if h_up in hash_dict_up:
        hash_dict_up[h_up]['orbit'].add(v)
    else:
        hash_dict_up[h_up] = {
            'color': free_color,
            'orbit': {v}
        }
        free_color += 1
    
    return hash_dict_up, free_color

# Update queue - basic version
def update_queue_add_all_neighbors(q, G, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    q_new = set(q)
    for v in q:
        q_new.update(G.neighbors(v))
    return q_new, coloring_up, hash_dict_up

# Update queue - remove asymmetric vertices

def update_queue_remove_asymmetric(q, G, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    q_new = set()
    asymmetric_vertices = set()
    for v in q:
        h_old, h_up = get_hash_pair(v, coloring_old, coloring_up)
        orbit_old = hash_dict_old[h_old]['orbit']
        orbit_up = hash_dict_up[h_up]['orbit']
        if len(orbit_old) == 1 and len(orbit_up) == 1:
            asymmetric_vertices.add(v)
        else:
            q_new.add(v)
            q_new.update(G.neighbors(v))
    recolor_unchanged_orbits(asymmetric_vertices, coloring_old, coloring_up, hash_dict_old, hash_dict_up)
            
    return q_new, coloring_up, hash_dict_up

def update_queue_remove_unchanged_orbits(q, G, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    unchanged_orbits = get_unchanged_orbits(q, coloring_old, coloring_up, hash_dict_old, hash_dict_up)
    recolor_unchanged_orbits(unchanged_orbits, coloring_old, coloring_up, hash_dict_old, hash_dict_up)
    
    q_new = set()
    affected_vertices = q.difference(unchanged_orbits)
    for v in affected_vertices:
        q_new.add(v)
        q_new.update(G.neighbors(v))

    return q_new, coloring_up, hash_dict_up

def get_unchanged_orbits(q, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    unchanged_orbits = set()
    seen_orbits = set()

    for v in q:
        h_old, h_up = get_hash_pair(v, coloring_old, coloring_up)
        if h_old in seen_orbits or h_up in seen_orbits:
            continue

        orbit_old = hash_dict_old[h_old]['orbit']
        orbit_up = hash_dict_up[h_up]['orbit']

        if orbit_up == orbit_old and orbit_up.issubset(q):
            unchanged_orbits.update(orbit_up)

        seen_orbits.update([h_old, h_up])

    return unchanged_orbits

def recolor_unchanged_orbits(unchanged_orbits, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    processed_hashes = set()

    for v in unchanged_orbits:
        h_old, h_up = get_hash_pair(v, coloring_old, coloring_up)

        # Revert color and hash
        coloring_up.at[v, 'color'] = coloring_old.at[v, 'color']
        coloring_up.at[v, 'hash'] = coloring_old.at[v, 'hash']

        # Update hash_dict if this hash hasn't been processed
        if h_up in hash_dict_up and h_up not in processed_hashes:
            orbit = hash_dict_up[h_up]['orbit']
            del hash_dict_up[h_up]

            if h_old not in hash_dict_up:
                hash_dict_up[h_old] = {
                    'color': coloring_old.at[v, 'color'],
                    'orbit': set()
                }
            hash_dict_up[h_old]['orbit'].update(orbit)
            processed_hashes.add(h_up)


def continue_color_refinement(G, c, m_pre, m_aft):
    colorings_up_list = []
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0,n)]) 
    
    while m_pre != m_aft:
        Ac = A@c.apply(log_pi)
        c, m = colors(c + Ac)
        colorings_up_list.append(pd.DataFrame({'color': c}))
        m_pre = m_aft
        m_aft = m
    return colorings_up_list


# Add new vertices to the queue
'''def update_queue(q, G, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    unchanged_orbits = set()
    seen_vertices = set()
    for v in q:
        if v not in seen_vertices:
            h_old, h_up = get_hash_pair(v, coloring_old, coloring_up)
            orbit_old = hash_dict_old[h_old]['orbit']
            orbit_up = hash_dict_up[h_up]['orbit']
            if orbit_up.issubset(q) and orbit_up == orbit_old:
                unchanged_orbits.update(orbit_up)
            seen_vertices.update(q.intersection( orbit_old.union(orbit_up) ))
    
    affected_vertices = q.difference(unchanged_orbits)
    affected_neighbors = set()
    for v in affected_vertices:
        affected_neighbors.update(G.neighbors(v))
    q_new = affected_vertices.union(affected_neighbors)
    return q_new
    
'''

'''
def update_queue(q, G, coloring_old, coloring_up, hash_dict_old, hash_dict_up):
    extended_q = set(q) | {u for v in q for u in G.neighbors(v)}
    print(f"Extended queue: {extended_q}")
    unchanged_orbits = set()
    seen_orbits = set()
    for v in q:
        h_old, h_up = get_hash_pair(v, coloring_old, coloring_up)
        if h_old in seen_orbits or h_up in seen_orbits:
            continue  # already checked

        orbit_old = hash_dict_old[h_old]['orbit']
        orbit_up = hash_dict_up[h_up]['orbit']

        if orbit_up == orbit_old and orbit_up.issubset(extended_q):
            unchanged_orbits.update(orbit_up)

        seen_orbits.update([h_old, h_up])
    print(f"Unchanged orbits: {unchanged_orbits}")
    q_new = extended_q - unchanged_orbits
    print(f"Updated queue: {q_new}")
    return q_new
'''