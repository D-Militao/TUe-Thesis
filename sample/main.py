from pprint import pprint
from time import time, strftime, gmtime
from functools import partial
import math
import pandas as pd

import snap

from constants import Constants
from graph_merge_summary import GraphMergeSummary
import estimation
from all_distances_sketches import GraphSketch
import util
import load_data

__start_time__ = time()


def elapsed_time_str(start_time):
    return strftime("%H:%M:%S", gmtime(time() - start_time))


def elapsed_time(start_time):
    return (time() - start_time)


def test_graph_merge_summary(network):
    summary = GraphMergeSummary(network)
    print(f"--> {elapsed_time_str(__start_time__)} Building evaluation network...")
    summary.build_evalutation_network()
    print(f"--> {elapsed_time_str(__start_time__)} Evaluation completed.")
    print(f"\t+++ Number of super nodes: {summary.evaluation_network.GetNodes()}")
    print(f"\t+++ Number of super edges: {summary.evaluation_network.GetEdges()}")
    print(f"--> {elapsed_time_str(__start_time__)} Building merge network...")
    summary.build_merge_network(is_target_merge=False)
    print(f"--> {elapsed_time_str(__start_time__)} Merging completed.")
    print(f"\t+++ Number of hyper nodes: {summary.merge_network.GetNodes()}")
    print(f"\t+++ Number of hyper edges: {summary.merge_network.GetEdges()}")


def test_ads(network, k):
    start_sketch = time()
    print(f"--> {elapsed_time(__start_time__)} Creating sketch...")
    graph_sketch = GraphSketch(network, k)
    graph_sketch.calculate_graph_sketch()
    print(f"--> {elapsed_time(__start_time__)} Sketch created.")
    end_sketch = time()
    
    # print(f"--> {elapsed_time(__start_time__)} Calculating neighborhood...")
    # graph_sketch.calculate_neighborhoods()
    # print(f"--> {elapsed_time(__start_time__)} Neighborhood calculated.")
    # tc_estimate_neighborhood = graph_sketch.estimate_cardinality_neighborhood(node_id)
    
    node_id = 939
    tc_estimate = graph_sketch.cardinality_estimation_node_id(node_id)
    
    bfs_tree = network.GetBfsTree(node_id, True, False)
    tc = bfs_tree.GetNodes()


def full_test_network_ads(network, node_ids, general_data, tc_data):
    k_values = [16, 32, 64, 128]
    N = node_ids.Len()
    for k in k_values:
        label_aux = f'ADS k={k}'
        start_sketch = time()
        graph_sketch = GraphSketch(network, k)
        graph_sketch.calculate_graph_sketch()
        sketch_creation_time = elapsed_time(start_sketch)

        tc_estimates = [] # snap.TFltV()
        node_est_times = [] # snap.TFltV()
        start_est = time()
        
        for node_id in node_ids:
            start_node_est = time()
            tc_estimates.append(
                graph_sketch.cardinality_estimation_node_id(node_id))
            node_est_times.append(elapsed_time(start_node_est))
        
        total_estimation_time = elapsed_time(start_est)
        tc_data[f'{label_aux} estimate'] = tc_estimates
        tc_data[f'{label_aux} estimate time'] = node_est_times
        
        general_data.setdefault(
            f'{label_aux} creation time', []).append( # snap.TFltV()
                sketch_creation_time)
        general_data.setdefault(
            f'{label_aux} total estimation time N={N}', []).append( # snap.TFltV()
                total_estimation_time)


def full_test_network_summary(network, node_ids, general_data, tc_data):
    summary = GraphMergeSummary(network)
    summary.build_evalutation_network()
    pass


def full_test_network(network, general_data, seed, N):
    tc_data = {}
    node_ids = snap.TIntV()
    
    rnd = snap.TRnd(seed)
    for i in range(N):
        node_ids.append(network.GetRndNId(rnd))
    tc_data['Node_ids'] = node_ids

    tcs = snap.TIntV()
    for node_id in node_ids:
        bfs_tree = network.GetBfsTree(node_id, True, False)
        tc = bfs_tree.GetNodes()
        tcs.append(tc)
    tc_data['TC'] = tcs

    full_test_network_ads(network, node_ids, general_data, tc_data)
    full_test_network_summary(network, node_ids, general_data, tc_data)
    return tc_data


def full_test(seed, N=1000):
    general_data = {}
    network_names_methods = {
        # 'shop_1k': partial(load_data.make_gmark_network,load_data.GMarkUseCase.shop, size=1000),
        # 'shop_5k': partial(load_data.make_gmark_network,load_data.GMarkUseCase.shop, size=5000),
        'shop_25k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=25000),
        # 'shop_50k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=50000),
        # 'shop_100k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=100000),
        # 'shop_200k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=200000),
        # 'shop_250k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=250000)
    }
        
    for network_name, network_method in network_names_methods.items():
        network = network_method()
        general_data.setdefault('Network', snap.TStrV()).append(network_name)
        tc_data = full_test_network(network, general_data, seed, N)
    
    general_data_df = pd.DataFrame.from_dict(general_data)
    tc_data_df = pd.DataFrame.from_dict(tc_data)

    general_data_df.to_csv('results/general_data.csv',index=False)
    tc_data_df.to_csv('results/tc_data.csv', index=False)


if __name__ == '__main__':
    # full_test(42, N=100)

    print(f"--> {elapsed_time_str(__start_time__)} Loading dataset...")
    # network = load_data.make_pg_paper_network()
    network = load_data.make_gmark_network(load_data.GMarkUseCase.shop, size=5000)
    print(f"--> {elapsed_time_str(__start_time__)} Dataset loaded.")
    print(f"\t+++ Number of Nodes: {network.GetNodes()}")
    print(f"\t+++ Number of Edges: {network.GetEdges()}")
    
    test_graph_merge_summary(network)

    # result = estimation.estimation_function_test(network, 42)
    # result = estimation.estimation_function_test_paper(42)
    # print(result)

    # test_ads(network, 30)

    
    
