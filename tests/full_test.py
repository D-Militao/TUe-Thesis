from time import time, strftime, gmtime
from functools import partial
import os
import pandas as pd

import snap

from .context import snap_util, load_data
from .context import GraphMergeSummary
from .context import GraphSketch
from .context import estimation_function

__start_time__ = time()


def elapsed_time_str(start_time: float) -> str:
    """Returns the elapsed time since the given start time as a string."""
    return strftime("%H:%M:%S", gmtime(time() - start_time))


def elapsed_time(start_time) -> float:
    """Returns the elapsed time since the given start time."""
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
    snap_util.print_all_edge_attributes(summary.merge_network)
    snap_util.print_all_node_attributes(summary.merge_network)


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
        print(f'{[elapsed_time_str(__start_time__)]} Creating sketches for k={k}...')
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
            f'{label_aux} creation time', []).append(sketch_creation_time) # snap.TFltV()
        general_data.setdefault(
            f'{label_aux} total estimation time N={N}', []).append(
                total_estimation_time) # snap.TFltV()


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


def full_test_network(network, general_data, seed, N):
    tc_results = {}
    node_ids = snap.TIntV()
    
    rnd = snap.TRnd(seed)
    for i in range(N):
        node_ids.append(network.GetRndNId(rnd))
    tc_results['Node_ids'] = node_ids

    tc_values = snap.TIntV()
    node_est_times = snap.TFltV()
    for node_id in node_ids:
        start_tc = time()
        bfs_tree = network.GetBfsTree(node_id, True, False)
        tc_values.append(bfs_tree.GetNodes())
        node_est_times.append(elapsed_time(start_tc))
        
    tc_results['TC'] = tc_values
    tc_results['TC time'] = node_est_times

    full_test_network_ads(network, node_ids, general_data, tc_results)
    # full_test_network_summary(network, node_ids, general_data, tc_data)
    return tc_results


def full_test(seed, N=1000):
    general_results = {}
    general_results['Network'] = snap.TStrV()
    network_names_methods = {
        # 'pg_paper': partial(load_data.make_pg_paper_network),
        'shop_1k': partial(load_data.load_gmark_network,load_data.GMarkUseCase.shop, size=1000),
        # 'shop_5k': partial(load_data.make_gmark_network,load_data.GMarkUseCase.shop, size=5000),
        # 'shop_25k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=25000),
        # 'shop_50k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=50000),
        # 'shop_100k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=100000),
        # 'shop_200k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=200000),
        # 'shop_250k': partial(load_data.make_gmark_network, load_data.GMarkUseCase.shop, size=250000)
    }
        
    # for network_name, load_network_method in network_names_methods.items():
    #     network = load_network_method()
    #     general_data.setdefault('Network', snap.TStrV()).append(network_name)
    #     tc_data = full_test_network(network, general_data, seed, N)
    
    files_sizes = {}
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                size = os.stat(path).st_size
                files_sizes[file] = size

    for file, size in sorted(files_sizes.items(), key=lambda item: item[1]):            
        general_results['Network'].append(file)
        filepath = os.path.join(root, file)

        start_time = time()
        print(f"{[elapsed_time_str(__start_time__)]} Loading {filepath}...")
        network = load_data.load_unlabeled_edge_file(filepath)
        print(f"{[elapsed_time_str(__start_time__)]} Loaded in: {elapsed_time_str(start_time)}")
        print(f"{[elapsed_time_str(__start_time__)]} Nodes: {network.GetNodes()}, Edges: {network.GetEdges()}")

        tc_results = full_test_network(network, general_results, seed, N)
        tc_results_df = pd.DataFrame.from_dict(tc_results)
        tc_results_df.to_csv(f"results/{strftime('%Y-%m-%d_%Hh%Mm%Ss', gmtime(time()))}_tc_results.csv", index=False)
        break
    
    general_results_df = pd.DataFrame.from_dict(general_results)
    general_results_df.to_csv(f"results/{strftime('%Y-%m-%d_%Hh%Mm%Ss', gmtime(time()))}_general_results.csv",index=False)