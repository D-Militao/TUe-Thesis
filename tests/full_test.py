from time import time, strftime, gmtime
from functools import partial
import os
import pandas as pd
import tracemalloc

import snap

from .context import snap_util, load_data
from .context import GraphMergeSummary
from .context import GraphSketch
from .context import estimation_function
from .test_util import elapsed_time, elapsed_time_str, stopwatch
from .test_all_distance_sketch import test_time_all_distance_sketch_node_ids


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
    snap_util.print_all_edge_attributes(summary.merge_network)
    snap_util.print_all_node_attributes(summary.merge_network)





def full_test_network_summary(network, node_ids, general_data, tc_data):
    merge_types = [False, True]
    N = node_ids.Len()

    start_init = time()
    summary = GraphMergeSummary(network)
    init_time = elapsed_time(start_init)
        
    start_eval = time()
    summary.build_evalutation_network()
    eval_time = elapsed_time(start_eval)
    print(f"\t+++ Number of super nodes: {summary.evaluation_network.GetNodes()}")
    print(f"\t+++ Number of super edges: {summary.evaluation_network.GetEdges()}")

    general_data.setdefault(f'Summary init time', []).append(init_time) # snap.TFltV()
    general_data.setdefault(f'Summary evaluation time', []).append(eval_time) # snap.TFltV()

    for merge_type in merge_types:
        label_aux = f'Summary is_target_merge={merge_type}'
        start_merge = time()
        summary.build_merge_network(is_target_merge=merge_types)
        merge_time = elapsed_time(start_merge)
        print(f"\t+++ Number of hyper nodes: {summary.merge_network.GetNodes()}")
        print(f"\t+++ Number of hyper edges: {summary.merge_network.GetEdges()}")

        tc_estimates = [] # snap.TFltV()
        node_est_times = [] # snap.TFltV()
        start_est = time()

        for node_id in node_ids:
            start_node_est = time()
            tc_estimates.append(
                summary.cardinality_estimation_node_id(node_id))
            node_est_times.append(elapsed_time(start_node_est))

        total_estimation_time = elapsed_time(start_est)
        tc_data[f'{label_aux} estimate'] = tc_estimates
        tc_data[f'{label_aux} estimate time'] = node_est_times

        general_data.setdefault(f'{label_aux} merge time', []).append(merge_time) # snap.TFltV()
        general_data.setdefault(
            f'{label_aux} total estimation time N={N}', []).append(total_estimation_time) # snap.TFltV()
        break


def full_test_memory_network(network, k_values: list, seed: int, N: int, network_results: dict) -> pd.DataFrame:
    node_results = pd.DataFrame()
    node_ids = snap.TIntV()
    
    rnd = snap.TRnd(seed)
    # rnd.Randomize() # Omit this line to get the same return values for different program executions 
    for i in range(N):
        node_ids.append(network.GetRndNId(rnd))
    node_results['Node ids'] = node_ids

    print(f'[{stopwatch()}] Calculating TC for N={node_ids.Len()} nodes.')
    tc_values = snap.TIntV()
    for node_id in node_ids:
        bfs_tree = network.GetBfsTree(node_id, True, False)
        tc_values.append(bfs_tree.GetNodes())
    node_results['TC'] = tc_values
    network_results.setdefault(f"TC time", []).append(total_tc_time)
    
    for k in k_values:
        print(f'[{stopwatch()}] Testing sketch for k={k}.')
        sketch_construction_time, total_estimation_time, estimates = test_time_all_distance_sketch_node_ids(network, node_ids, k)
        node_results[f"ADS k={k}"] = estimates
        network_results.setdefault(f"ADS k={k} memory size", []).append(sketch_construction_time)
        network_results.setdefault(f"ADS k={k} total estimation time", []).append(total_estimation_time)
    
    # full_test_network_summary(network, node_ids, general_data, tc_data)
    return node_results


def full_test_memory(seed, N, k_values, files_sizes, root):
    test_timestamp = strftime('%Y-%m-%d_%Hh%Mm%Ss', gmtime(time()))
    time_results = {}
    time_results['Network'] = []
    time_results['#Nodes'] = []
    time_results['#Edges'] = []
    time_results['Size'] = []

    tracemalloc.start()
    for file, size in sorted(files_sizes.items(), key=lambda item: item[1]):
        filepath = os.path.join(root, file)

        before_loading, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        print(f"[{stopwatch()}] Loading {filepath}...")
        network = load_data.load_unlabeled_edge_file(filepath)
        loading_time = elapsed_time(start_time)
        no_nodes = network.GetNodes()
        no_edges = network.GetEdges()
        print(f"[{stopwatch()}] Loaded in: {elapsed_time_str(start_time)}")
        print(f"[{stopwatch()}] Nodes: {no_nodes}, Edges: {no_edges}")
        
        time_results['Network'].append(file)
        time_results['Loading time'].append(loading_time)
        time_results['#Nodes'].append(no_nodes)
        time_results['#Edges'].append(no_edges)

        node_results = full_test_time_network(network, k_values, seed, N, time_results)

        node_results.to_csv(f"results/{test_timestamp}_node_results_N={N}.csv", index=False)
        break
    
    network_results_df = pd.DataFrame.from_dict(time_results)
    network_results_df.to_csv(f"results/{test_timestamp}_time_results_N={N}.csv", index=False)


def full_test_time_network(network, k_values: list, seed: int, N: int, time_results: dict) -> pd.DataFrame:
    node_results = pd.DataFrame()
    node_ids = snap.TIntV()
    
    rnd = snap.TRnd(seed)
    # rnd.Randomize() # Omit this line to get the same return values for different program executions 
    for i in range(N):
        node_ids.append(network.GetRndNId(rnd))
    node_results['Node ids'] = node_ids

    print(f'[{stopwatch()}] Calculating TC for N={node_ids.Len()} nodes.')
    tc_values = snap.TIntV()
    start_tc = time()
    for node_id in node_ids:
        bfs_tree = network.GetBfsTree(node_id, True, False)
        tc_values.append(bfs_tree.GetNodes())
    total_tc_time = elapsed_time(start_tc)
    node_results['TC'] = tc_values
    time_results.setdefault(f"TC time", []).append(total_tc_time)
    
    for k in k_values:
        print(f'[{stopwatch()}] Testing sketch for k={k}.')
        sketch_construction_time, total_estimation_time, estimates = test_time_all_distance_sketch_node_ids(network, node_ids, k)
        node_results[f"ADS k={k}"] = estimates
        time_results.setdefault(f"ADS k={k} construction time", []).append(sketch_construction_time)
        time_results.setdefault(f"ADS k={k} total estimation time", []).append(total_estimation_time)
    
    # full_test_network_summary(network, node_ids, general_data, tc_data)
    return node_results


def full_test_time(seed, N, k_values, files_sizes, root):
    test_timestamp = strftime('%Y-%m-%d_%Hh%Mm%Ss', gmtime(time()))
    time_results = {}
    time_results['Network'] = []
    time_results['#Nodes'] = []
    time_results['#Edges'] = []
    time_results['Loading time'] = []

    for file, size in sorted(files_sizes.items(), key=lambda item: item[1]):
        filepath = os.path.join(root, file)

        start_time = time()
        print(f"[{stopwatch()}] Loading {filepath}...")
        network = load_data.load_unlabeled_edge_file(filepath)
        loading_time = elapsed_time(start_time)
        no_nodes = network.GetNodes()
        no_edges = network.GetEdges()
        print(f"[{stopwatch()}] Loaded in: {elapsed_time_str(start_time)}")
        print(f"[{stopwatch()}] Nodes: {no_nodes}, Edges: {no_edges}")
        
        time_results['Network'].append(file)
        time_results['Loading time'].append(loading_time)
        time_results['#Nodes'].append(no_nodes)
        time_results['#Edges'].append(no_edges)

        node_results = full_test_time_network(network, k_values, seed, N, time_results)

        node_results.to_csv(f"results/{test_timestamp}_node_results_N={N}.csv", index=False)
        break
    
    network_results_df = pd.DataFrame.from_dict(time_results)
    network_results_df.to_csv(f"results/{test_timestamp}_time_results_N={N}.csv", index=False)


def full_test(seed, N, k_values=[5, 10, 50, 100]):
    # Get the data files and their sizes in the data directory 
    files_sizes = {}
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                size = os.stat(path).st_size
                files_sizes[file] = size
    
    # We need them separate because the memory tests cause a slow down of ~30%
    full_test_time(seed, N, k_values, files_sizes, root)
    full_test_memory(seed, N, k_values, files_sizes, root)

    




# network_names_methods = {
    # 'pg_paper': partial(load_data.make_pg_paper_network),
    # 'shop_1k': partial(load_data.load_gmark_network,load_data.GMarkUseCase.shop, size=1000),
    # 'shop_5k': partial(load_data.make_gmark_network,load_data.GMarkUseCase.shop, size=5000),
    # 'shop_25k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=25000),
    # 'shop_50k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=50000),
    # 'shop_100k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=100000),
    # 'shop_200k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=200000),
    # 'shop_250k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=250000)
# }
    
# for network_name, load_network_method in network_names_methods.items():
#     network = load_network_method()
#     general_data.setdefault('Network', snap.TStrV()).append(network_name)
#     tc_data = full_test_network(network, general_data, seed, N)