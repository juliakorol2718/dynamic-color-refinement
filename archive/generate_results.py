import networkx as nx
import pandas as pd
import random
import numpy as np
from cup_gather_data import hcgcr_data
import json
from tqdm import tqdm
from pathlib import Path
from unpack_graphs import unpack_graphs
from cup_gather_data import Iteration
from cup_main_gather_queue_data import updateCR
from archive.cup_functions import update_queue_add_all_neighbors, \
    update_queue_remove_asymmetric, update_queue_remove_unchanged_orbits

base_path = Path(r'C:\Users\korol\OneDrive\Pulpit\Master\CUP - remove asymmetric, remove orbits, counting changes\result_graphs')
update_functions = [update_queue_add_all_neighbors, update_queue_remove_asymmetric, update_queue_remove_unchanged_orbits]
test_graphs = unpack_graphs()
##############################################################################################
# need different algorithms and different S size 
# size of S = 1%, 5%, 20% of edges  
def pair_generator(n): 
    used_pairs = set() 
    while True: 
        pair = random.sample(range(n), 2)
        pair = tuple(sorted(pair))
        if pair not in used_pairs: 
            used_pairs.add(pair) 
            yield pair 

def change_random_set_of_edges(G, s_len):
    G_new = G.copy()
    S = set()
    for _ in range(s_len):
        edge = next(pair_generator(len(G_new.nodes)))
        if G_new.has_edge(edge[0], edge[1]):
            G_new.remove_edge(edge[0], edge[1])
        else:
            G_new.add_edge(edge[0], edge[1])
        S.add(edge[0])
        S.add(edge[1])
    return G_new, S


def update_dict(G, S, iterations_up, q_lens):
    avg = (sum(q_lens) / len(q_lens)) if q_lens else 0.0
    data = {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "graph": json.dumps(list(G.edges())),
        "S": json.dumps(list(S)),
        "q_lens": json.dumps(q_lens),
        "q_len_average": avg,
        "coloring": json.dumps([
            {"color": it.coloring["color"].to_list(),
                "hash": it.coloring["hash"].to_list()}
            for it in iterations_up
        ]),
        "hash_dict": json.dumps([it.hash_dict for it in iterations_up], default=list)
    }
    return data

'''
name = next(iter(test_graphs))
row = test_graphs[name].iloc[0]
q_func = update_functions[0]
G = row["graph"]
iterations = row["iterations"]
s_len = int(np.floor(G.number_of_edges() * 0.05))
G_up, S = change_random_set_of_edges(G, s_len)
iterations_up, q_lens = updateCR(G_up, S, iterations, q_func)
data = update_dict(G_up, S, iterations_up, q_lens)
pd.DataFrame([data]).to_csv("test.csv", index=False)
'''


print("Start")
for s_p, folname in zip([0.01, 0.05, 0.2], ["one_percent", "five_percent", "twenty_percent"]):
    print("s_p = ", s_p)
    path = base_path / folname
    for q_func in update_functions:
        print(q_func)
        for name, df in test_graphs.items():
            print(name)
            result = []
            for row in tqdm(df.itertuples(index=False), desc="Generating for the file: "):
                G = row.graph
                iterations = row.iterations
                s_len = int(np.floor(G.number_of_edges())*s_p)
                G_up, S = change_random_set_of_edges(G, s_len)
                iterations_up, q_lens = updateCR(G, S, iterations, q_func)
                result.append(update_dict(G_up, S, iterations_up, q_lens))
            dfres = pd.DataFrame(result)
            dfres.to_csv(path / f"result_{name}", index=False)
print("end")
