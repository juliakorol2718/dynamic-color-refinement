import numpy as np
from sympy import prime as pi
from tqdm import tqdm

PRECISION = 7

def build_log_pi_table(max_color, pi_func, precision=PRECISION):
    table = np.empty(max_color + 1, dtype=np.float64)
    table[0] = -np.inf  # unused if colors start at 1
    for c in tqdm(range(1, max_color + 1), desc="Generating hashes"):
        table[c] = np.log(float(pi_func(int(c))))
    return np.round(table, precision)

max_color = 2000
#table = build_log_pi_table(max_color, pi)
#np.save("log_pi_table.npy", table)


# for epinions:
table = build_log_pi_table(75879, pi)
np.save("log_pi_table_big.npy", table)
