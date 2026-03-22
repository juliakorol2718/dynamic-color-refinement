import numpy as np
import networkx as nx
from hcgcr_gather_data import build_color_partition
from cup_classes import Iteration
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PI_TABLE = np.load(PROJECT_ROOT / "log_pi_table.npy")
PRECISION = 7


def update_hash_value(q_list: list, A: np.ndarray, hash_old: np.ndarray, it_prev: Iteration):
    """
    Recompute hash values for vertices in ``q_list``.

    The hashes are updated using the same rule as in ``hcgcr_data``,
    combining the current vertex color with an aggregation of transformed
    neighbor colors. Only vertices in ``q_list`` are recomputed, while the
    remaining hash values are copied unchanged.
    
    Args
        q_list (list[int]):
            Current queue.
        hash_old (np.ndarray):
            Array with hashes assinged to vertices in the original coloring.
        it_prev (Iteration) (np.ndarray):
            Previous iteration in from the update process.

    Returns:
        np.ndarray
            Array with the updated values of hashes.
    """
    if not q_list:
        return hash_old.copy()

    color_prev = it_prev.color
    c_log_pi = LOG_PI_TABLE[color_prev]

    Ac_upd_q = A[q_list, :] @ c_log_pi
    Ac_upd_q = np.asarray(Ac_upd_q).ravel()

    hash_upd_q = np.round(color_prev[q_list] + Ac_upd_q, PRECISION)

    hash_upd = hash_old.copy()
    hash_upd[q_list] = hash_upd_q
    return hash_upd


def update_colors(q_list: list, it_old: Iteration, hash_upd: np.ndarray):
    """
    Update vertex colors after hash values change for a subset of vertices.

    The function updates the color partition induced by the hash values,
    modifying only the vertices in ``q``. Vertices are removed from the
    color classes corresponding to their old hashes and inserted into the
    classes determined by ``hash_upd``. New color classes are created if
    necessary.

    Args
        q_list (list[int]):
            Current queue.
        it_old (Iteration):
            Original iteration containing the old coloring, hashes, and hash
            dictionary.
        hash_upd (np.ndarray):
            Array of updated hash values for all vertices.

    Returns:
        Iteration
            New iteration object containing the updated colors, hashes,
            and hash partition.
    """
    if not q_list:
        return it_old.copy()
    
    color_old = it_old.color
    hash_old = it_old.hash
    hash_dict_old = it_old.hash_dict

    color_upd = color_old.copy()

    h_old_q = hash_old[q_list]
    h_upd_q  = hash_upd[q_list]
    
    hash_dict_upd = {
        h: {
            "color": data["color"], 
            "orbit": set(data["orbit"])}
        for h, data in hash_dict_old.items()
    }
    free_color = len(hash_dict_upd) + 1
    
    # remove old hashes
    for v, h_old in zip(q_list, h_old_q):
        orbit_old = hash_dict_upd[h_old]['orbit']
        orbit_old.discard(v)
        
        if len(orbit_old) == 0:
            del hash_dict_upd[h_old]
    
    # add new hashes
    for v, h_upd in zip(q_list, h_upd_q):
        if h_upd in hash_dict_upd:
            hash_dict_upd[h_upd]['orbit'].add(v)
        else:
            hash_dict_upd[h_upd] = {
                'color': free_color,
                'orbit': {v}
            }
            free_color += 1
            
        color_upd[v] = hash_dict_upd[h_upd]['color']
        
    return Iteration(
        color=color_upd, 
        hash=hash_upd, 
        hash_dict=hash_dict_upd
    )


#######################################################################################################
# UPDATE FUNCTIONS
#######################################################################################################

# Update queue - add all neighbors (CUP)

def update_queue_cup(q_list: list, G: nx.graph, it_upd: Iteration, _):
    """CUP: Queue update strategy that adds the neighbors of all vertices in q_list to the queue."""
    adj = G.adj
    q_new = set(q_list)
    for v in q_list:
        q_new.add(v)
        q_new.update(adj[v])
    return q_new, it_upd


#######################################################################################################

# Update queue - remove asymmetric vertices (CAS)

def update_queue_cas(q_list: list, G: nx.graph, it_upd: Iteration, it_old: Iteration):
    """
    CAS: For each vertex in ``q_list``, the function compares its old and
    updated orbit. If both the old and new orbit contain only
    that vertex, the vertex is treated as asymmetric and removed from further
    propagation. Otherwise, the vertex and its neighbors are added to the
    next queue.

    Asymmetric vertices are recolored to their old color using ``recolor_unchanged_orbits``.
    """
    q_new = set()
    asymmetric_vertices = set()
    adj = G.adj

    h_old_q = it_old.hash[q_list]
    h_upd_q  = it_upd.hash[q_list]

    for v, h_old, h_upd in zip(q_list, h_old_q, h_upd_q):
        orbit_old = it_old.hash_dict[h_old]["orbit"]
        orbit_upd  = it_upd.hash_dict[h_upd]["orbit"]
        if len(orbit_old) == 1 and len(orbit_upd) == 1:
            asymmetric_vertices.add(v)
        else:
            q_new.add(v)
            q_new.update(adj[v])
    q_new.difference_update(asymmetric_vertices)

    recolor_unchanged_orbits(asymmetric_vertices, it_upd, it_old)
    
    return q_new, it_upd


#######################################################################################################

# Update queue - remove unchanged orbits (COR)

def update_queue_cor(q_list: list, G: nx.graph, it_upd: Iteration, it_old: Iteration):
    """
    COR: For each vertex in ``q_list``, the function compares its old and
    updated orbit. Vertices whose orbit did not change are treated as
    unchanged and excluded from further propagation. Otherwise, the vertex
    and its neighbors are added to the next queue. Additionally, from the 
    new queue are removed the unchanged asymmetric vertices, following the logic 
    of the CAS function.

    Vertices in unchanged orbits are recolored to their old color using
    ``recolor_unchanged_orbits``.
    """
    unchanged_orbits, asymmetric_vertices = get_unchanged_orbits(q_list, it_upd, it_old)
    recolor_unchanged_orbits(unchanged_orbits, it_upd, it_old)
    
    q_new = set()
    adj = G.adj
    affected_vertices = set(q_list).difference(unchanged_orbits)
    for v in affected_vertices:
        q_new.add(v)
        q_new.update(adj[v])
        
    q_new.difference_update(asymmetric_vertices)
    
    return q_new, it_upd

#######################################################################################################

# Helper functions

def get_unchanged_orbits(q: set, it_upd: Iteration, it_old: Iteration):
    """
    Finds unchanged orbits among vertices in ``q``.

    For each vertex in ``q``, the old and updated orbit are compared. If the
    orbits are identical and fully contained in ``q``, the orbit is marked
    as unchanged. Unchanged orbits with only one vertex are additionally 
    classified as asymmetric.

    Returns the vertices in unchanged orbits and the asymmetric vertices (subset of the first).
    """
    unchanged_orbits = set()
    seen_vertices = set()
    asymmetric_vertices = set()

    for v in q:
        if v in seen_vertices:
            continue
        
        h_old = it_old.hash[v]
        h_upd  = it_upd.hash[v]

        orbit_old = it_old.hash_dict[h_old]['orbit']
        orbit_upd  = it_upd.hash_dict[h_upd]['orbit']
        
        # Mark vertices from old and new orbits as visited
        seen_vertices.update(orbit_old)
        seen_vertices.update(orbit_upd)

        if orbit_upd.issubset(q) and orbit_upd == orbit_old:
            unchanged_orbits.update(orbit_old)
            if len(orbit_upd) == 1:
                asymmetric_vertices.add(v)

    return unchanged_orbits, asymmetric_vertices

def recolor_unchanged_orbits(unchanged_orbits: set, it_upd: Iteration, it_old: Iteration):
    """
    For vertices in unchanged orbits, restore the color and hash values in
    the updated iteration to those from the old iteration.
    """
    
    if not unchanged_orbits:
        return
    
    unchanged = list(unchanged_orbits)
    
    hash_dict_upd = it_upd.hash_dict
    
    for v in unchanged:
        h_old = it_old.hash[v]
        h_upd = it_upd.hash[v]

        # Update hash_dict if this hash hasn't been processed
        if h_upd in hash_dict_upd:
            orbit = hash_dict_upd[h_upd]['orbit']
            del hash_dict_upd[h_upd]

            if h_old not in hash_dict_upd:
                hash_dict_upd[h_old] = {
                    'color': it_old.color[v],
                    'orbit': set()
                }
            hash_dict_upd[h_old]['orbit'].update(orbit)
    
    # revert color and hash:
    it_upd.color[unchanged] = it_old.color[unchanged]
    it_upd.hash[unchanged] = it_old.hash[unchanged]

def continue_color_refinement(G: nx.graph, c: np.ndarray, m_pre: int, m_aft: int):
    """Given a current coloring of the graph, return a list of further steps
    of color refinement algorithm."""
    colorings_upd_list = []
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0,n)]) 

    while m_pre != m_aft:
        c_log_pi = LOG_PI_TABLE[c]
        Ac = A @ c_log_pi
        hash = np.round(c + Ac, PRECISION)
        c, hash_dict = build_color_partition(hash)
    
        m_pre = m_aft
        m_aft = len(hash_dict)
        colorings_upd_list.append(c)
            
    return colorings_upd_list


if __name__ == "__main__":
    print("this is main")
