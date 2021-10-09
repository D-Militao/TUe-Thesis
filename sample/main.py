from pprint import pprint
import time
import math
import pandas as pd

import snap

from constants import Constants
from graph_merge_summary import GraphMergeSummary
import estimation
import all_distances_sketches
import util
import load_data

__start_time__ = time.time()


def elapsed_time(__start_time__):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - __start_time__))


def test_graph_merge_summary(network):
    summary = GraphMergeSummary(network)
    print(f"--> {elapsed_time(__start_time__)} Building evaluation network...")
    summary.build_evalutation_network()
    print(f"--> {elapsed_time(__start_time__)} Evaluation completed.")
    print(f"\t+++ Number of super nodes: {summary.evaluation_network.GetNodes()}")
    print(f"\t+++ Number of super edges: {summary.evaluation_network.GetEdges()}")
    print(f"--> {elapsed_time(__start_time__)} Building merge network...")
    summary.build_merge_network(is_target_merge=False)
    print(f"--> {elapsed_time(__start_time__)} Merging completed.")
    print(f"\t+++ Number of hyper nodes: {summary.merge_network.GetNodes()}")
    print(f"\t+++ Number of hyper edges: {summary.merge_network.GetEdges()}")


def test_ads(network, k):
    print(f"--> {elapsed_time(__start_time__)} Creating sketch...")
    graph_sketch = all_distances_sketches.GraphSketch(network, k)
    graph_sketch.calculate_graph_sketch()
    print(f"--> {elapsed_time(__start_time__)} Sketch created.")
    
    print(f"--> {elapsed_time(__start_time__)} Calculating neighborhood...")
    graph_sketch.calculate_neighborhoods()
    print(f"--> {elapsed_time(__start_time__)} Neighborhood calculated.")
    
    node_id = 939
    tc_estimate = graph_sketch.estimate_cardinality(node_id)
    bfs_tree = network.GetBfsTree(node_id, True, False)
    tc = bfs_tree.GetNodes()
    print(tc, tc_estimate)
    # results = {}
    # x = 0
    # for node_id, neighborhood in graph_sketch.neighborhoods.items():
    #     tc_estimate = 0
    #     for dist, neighborhood_size in neighborhood.items():
    #         tc_estimate += neighborhood_size
    #     bfs_tree = network.GetBfsTree(node_id, True, False)
    #     tc = bfs_tree.GetNodes()
    #     results[node_id] = (tc, tc_estimate)
    #     x += 1
    #     if x == 20:
    #         break
    
    # pprint(results)




if __name__ == '__main__':
    print(f"--> {elapsed_time(__start_time__)} Loading dataset...")
    
    # network = load_data.make_pg_paper_network()
    network = load_data.make_gmark_network(load_data.GMarkUseCase.shop, size=25000)
    
    print(f"--> {elapsed_time(__start_time__)} Dataset loaded.")
    print(f"\t+++ Number of Nodes: {network.GetNodes()}")
    print(f"\t+++ Number of Edges: {network.GetEdges()}")
    
    # test_graph_merge_summary(network)

    # result = estimation.estimation_function_test(network, 42)
    # result = estimation.estimation_function_test_paper(42)
    # print(result)

    test_ads(network, 10)

    
    
