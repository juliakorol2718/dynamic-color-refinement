import networkx as nx
import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
base_path = PROJECT_ROOT / "experiment_new" / "test_graphs"

from cup_classes import Iteration

def nxgraph(n, edges):
    G = nx.Graph()
    G.add_edges_from([tuple(e) for e in edges])
    G.add_nodes_from(range(int(n)))
    return G

def restore_dict_values(hash_dict):
    output = {}
    for hash_str, val in hash_dict.items():
        
        hash_key = round(float(hash_str), 7)  # same precision as in json
        color = int(val["color"])
        orbit = val.get("orbit")
        
        if isinstance(orbit, list):
            orbit = set(orbit)

        output[hash_key] = {"color": color, "orbit": orbit}

    return output

def unpack(name):
    df = pd.read_csv(base_path / name)
    # dataframe with two columns: G - the graph, iterations with list of iteration objects
    for col in ["graph", "coloring", "hash_dict"]:
        df[col] = df[col].apply(json.loads)
    graphs = []
    iterations = []
    for _, row in df.iterrows():
        graphs.append(nxgraph(row["n"], row["graph"]))
        iter = []
        for coloring_dict, hash_dict_from_json in zip(row["coloring"], row["hash_dict"]):
            coloring = pd.DataFrame(coloring_dict)
            hash_dict = restore_dict_values(hash_dict_from_json)
            iter.append(IterationExperiment(coloring, hash_dict))
        iterations.append(iter)
    return pd.DataFrame({"graph": graphs, "iterations": iterations})

# unpack graphs from the files
def unpack_graphs():
    #limits = [(10,100), (150,300), (300,500)]
    limits = [(10,150)]
    p_list = [0.1, 0.3, 0.5]
    graph_type = ["tree", 0]
    test_graphs = {}
    for lim in limits:
        for g in graph_type:
            if g == "tree":
                fname = f"trees_{lim[0]}_{lim[1]}.csv"
                test_graphs[fname] = unpack(fname)
            else:
                for p in p_list:
                    fname = f"random_graphs_{lim[0]}_{lim[1]}_p_{p}.csv"
                    test_graphs[fname] = unpack(fname)
    return test_graphs

