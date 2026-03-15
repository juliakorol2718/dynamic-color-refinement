import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import typing

from cup_functions_faster import ColoringState, update_hash_value, update_colors, update_queue_add_all_neighbors, \
                                get_unchanged_orbits, recolor_unchanged_orbits, update_queue_remove_asymmetric, \
                                update_queue_remove_unchanged_orbits
from cup_main_faster import updateCR

from cup_gather_data import hcgcr_data, Iteration

LOG_PI_TABLE = np.load("log_pi_table.npy")
PRECISION = 7
# =============================================================================================================================================
# TEST HCGCR
# =============================================================================================================================================
class TestHcgcr(unittest.TestCase):
    G = nx.Graph([(0,4), (1,4), (2,3), (3,4)])
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0, 5)]).toarray()
    iterations = hcgcr_data(G)
    
    def test_stop_condition(self):
        I1 = self.iterations[-1]
        I2 = self.iterations[-2]
        self.assertEqual(I1.nr_of_colors, I2.nr_of_colors, "error with stop condition")
        
    def test_check_coloring(self):
        c_pre_list = [np.ones(5, dtype=int), np.array([1,1,1,2,3], dtype=int), np.array([1,1,2,3,4], dtype=int)]
        c_expected_list = [np.array([1,1,1,2,3]), np.array([1,1,2,3,4]), np.array([1,1,2,3,4], dtype=int)]
        
        for i, (c_pre, c_expected) in enumerate(zip(c_pre_list, c_expected_list)):
            c = np.array(self.iterations[i].coloring['color'])
            h = np.array(self.iterations[i].coloring['hash']).astype(str)
            
            Ac_expected = self.A @ LOG_PI_TABLE[c_pre]
            h_expected = np.round(c_pre + Ac_expected, PRECISION).astype(str)
            
            self.assertTrue(np.array_equal(c, c_expected),  f"wrong color {i}, got {c}, expected {c_expected}")
            self.assertTrue(np.array_equal(h, h_expected), f"wrong hash {i}, got {h}, expected {h_expected}")

# =============================================================================================================================================
# TEST UPDATE CR
# =============================================================================================================================================
class TestCUPFunctions(unittest.TestCase):
    G = nx.Graph([(0,4), (1,4), (2,3), (3,4)])
    A = nx.adjacency_matrix(G, nodelist = [i for i in range(0, 5)]).toarray()
    iterations = hcgcr_data(G)
    G_up = nx.Graph([(0,4), (1,4), (2,3), (3,4), (0,1)])
    A_up = nx.adjacency_matrix(G_up, nodelist = [i for i in range(0, 5)]).toarray()
    
    # define expected results
    q_list = [[0,1], [0,1,4], [0,1,3,4]]
    c_pre_list = [np.ones(5, dtype=int), np.array([2,2,1,2,3], dtype=int), np.array([5,5,2,3,6], dtype=int)]
    h_ex_list = []
    for c_pre in c_pre_list:
        Ac_expected = A_up @ LOG_PI_TABLE[c_pre]
        h_up_expected = np.round(c_pre + Ac_expected, PRECISION).astype(str)
        h_ex_list.append(h_up_expected)
    
    # UPDATE HASH
    def test_update_hash(self):
        # q, A, hash_old, c_pre
        for i, (q, c_pre, h_up_expected) in enumerate(zip(self.q_list, self.c_pre_list, self.h_ex_list)):
            h_old = np.array(self.iterations[i].coloring['hash']).astype(str)
            c_pre_df = pd.DataFrame({'color': c_pre})
            c_pre_df = c_pre_df.astype({"color": "int64"})
            h_up = update_hash_value(q, self.A_up, h_old, c_pre_df)
            self.assertTrue(np.array_equal(h_up, h_up_expected),  f"wrong h_up {i}, got {h_up}, expected {h_up_expected}")
    
    # UPDATE COLORS
    def test_update_colors(self):
        # q, state_old, hash_up
        for i, (q, h_up) in enumerate(zip(self.q_list, self.h_ex_list)):
            state_old = ColoringState(
                color= np.array(self.iterations[i].coloring['color']),
                hash = np.array(self.iterations[i].coloring['hash']).astype(str),
                hash_dict= self.iterations[i].hash_dict
            ) 
            state_up = update_colors(q, state_old, h_up)
            if i == 0:
                self.assertEqual(state_up.hash_dict[h_up[0]]["orbit"], set([0,1,3]), f"incorrect orbit in {state_up.hash_dict[h_up[0]]}")
                self.assertEqual(state_up.hash_dict[h_up[0]]["color"], 2, f"incorrect color in {state_up.hash_dict[h_up[0]]}")
            if i == 1:
                self.assertEqual(state_up.hash_dict[h_up[0]]["color"], 5, f"incorrect color in {state_up.hash_dict[h_up[0]]}")
                self.assertEqual(state_up.hash_dict[h_up[4]]["color"], 6, f"incorrect color in {state_up.hash_dict[h_up[4]]}")
                self.assertFalse(state_old.hash[0] in state_up.hash_dict, 
                                f"hash {state_old.hash[0]} of vertex 0 was not removed from the dict, {state_up.hash_dict}.")
    
    # QUEUE FUNCTIONS CUP
    def test_update_queue_add_all_neighbors(self):
        q1, _ = update_queue_add_all_neighbors(set([0,1]), self.G_up, None, None)
        q2, _ = update_queue_add_all_neighbors(set([0,1,4]), self.G_up, None, None)
        self.assertEqual(q1, set([0,1,4]), f"wrong queue in CUP")
        self.assertEqual(q2, set([0,1,4,3]), f"wrong queue in CUP")
        
    # GET UNCHANGED ORBITS,  RECOLOR UNCHANGED ORBITS
    def test_unchanged_orbits(self):
        q = [0,1,4]
        c_pre_df = pd.DataFrame({'color': self.c_pre_list[1]})
        c_pre_df = c_pre_df.astype({"color": "int64"})
        state_old = ColoringState(
            color= np.array(self.iterations[1].coloring['color']),
            hash = np.array(self.iterations[1].coloring['hash']).astype(str),
            hash_dict= self.iterations[1].hash_dict
        )
        h_up = update_hash_value(q, self.A_up, state_old.hash, c_pre_df)
        state_up = update_colors(q, state_old, h_up)
        # get unchanged orbits
        unchanged_orbits = get_unchanged_orbits(q, state_up, state_old)
        # recolor unchanged orbits
        recolor_unchanged_orbits(q, state_up, state_old)
        # tests
        self.assertEqual(unchanged_orbits, set(q), f"did not recognize unchanged orbits {unchanged_orbits}")
        self.assertEqual(state_up.hash_dict[h_up[0]]["color"], 1, f"incorrect color in {state_up.hash_dict[h_up[0]]}")
        self.assertEqual(state_up.hash_dict[h_up[1]]["color"], 1, f"incorrect color in {state_up.hash_dict[h_up[0]]}")
        self.assertEqual(state_up.hash_dict[h_up[4]]["color"], 4, f"incorrect color in {state_up.hash_dict[h_up[0]]}")
    
    # QUEUE FUNCTIONS CAP
    def test_update_queue_remove_asymmetric(self):
        # definitions
        q = [0,1,4]
        c_pre_df = pd.DataFrame({'color': self.c_pre_list[1]})
        c_pre_df = c_pre_df.astype({"color": "int64"})
        state_old = ColoringState(
            color= np.array(self.iterations[1].coloring['color']),
            hash = np.array(self.iterations[1].coloring['hash']).astype(str),
            hash_dict= self.iterations[1].hash_dict
        )
        h_up = update_hash_value(q, self.A_up, state_old.hash, c_pre_df)
        state_up = update_colors(q, state_old, h_up)
        # result
        q_new, state_up = update_queue_remove_asymmetric(q, self.G_up, state_up, state_old)
        self.assertEqual(q_new, set([0,1]), 
                        f"wrong queue in CAP, q_new = {q_new}, h_dict_old = {state_old.hash_dict}, h_dict_new = {state_up.hash_dict}")
        self.assertEqual(state_up.color[4], 4, f"did not update color of an asym vertex, {state_up.color[4]}")
        
    # QUEUE FUNCTIONS REMOVE ORBITS
    def test_update_queue_remove_unchanged_orbits(self):
        # definitions
        q = [0,1,4]
        c_pre_df = pd.DataFrame({'color': self.c_pre_list[1]})
        c_pre_df = c_pre_df.astype({"color": "int64"})
        state_old = ColoringState(
            color= np.array(self.iterations[1].coloring['color']),
            hash = np.array(self.iterations[1].coloring['hash']).astype(str),
            hash_dict= self.iterations[1].hash_dict
        )
        h_up = update_hash_value(q, self.A_up, state_old.hash, c_pre_df)
        state_up = update_colors(q, state_old, h_up)
        # result
        q_new, state_up = update_queue_remove_unchanged_orbits(q, self.G_up, state_up, state_old)
        self.assertEqual(q_new, set(), 
                        f"wrong queue in unchanged orbits, q_new = {q_new}, h_dict_old = {state_old.hash_dict}, h_dict_new = {state_up.hash_dict}")
        self.assertEqual(state_up.color[0], 1, f"did not update color of an unchanged vertex, {state_up.color[0]}")

    # QUEUE FUNCTIONS MAIN
    def test_updateCR(self):
        update_functions = [update_queue_add_all_neighbors, update_queue_remove_asymmetric, update_queue_remove_unchanged_orbits]
        for i, u_func in enumerate(update_functions):
            colorings_up_list, q_len_list = updateCR(self.G_up, set([0,1]), self.iterations, u_func)
            col0 = colorings_up_list[-1][0]
            col4 = colorings_up_list[-1][4]
            if i == 0:
                self.assertEqual(q_len_list, [2,3,4], f"wrong q_len_list for CUP funct, {q_len_list}")
                self.assertEqual(col0, 5, f"wrong final color in CUP of node 0, {col0}")
                self.assertEqual(col4, 6, f"wrong final color in CUP of node 4, {col4}")
            if i == 1:
                self.assertEqual(q_len_list, [2,3,2], f"wrong q_len_list for CAP funct, {q_len_list}")
                self.assertEqual(col0, 5, f"wrong final color in CAP of node 0, {col0}")
                self.assertEqual(col4, 4, f"wrong final color in CAP of node 4, {col4}")
            if i == 2:
                self.assertEqual(q_len_list, [2,3,0], f"wrong q_len_list for COrbits funct, {q_len_list}")
                self.assertEqual(col0, 1, f"wrong final color in COrbits of node 0, {col0}")
                self.assertEqual(col4, 4, f"wrong final color in COrbits of node 4, {col4}")
                
###########################################################################################
if __name__ == "__main__":
    unittest.main(verbosity=2)