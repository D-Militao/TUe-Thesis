from math import inf
import os
import pandas as pd
import networkx as nx

import snap

from .context import snap_util, load_data
from .context import estimation_function
from .test_util import test_print, current_time_str, current_date_time_str, size_transitive_closure_node_ids, size_transitive_closure_node_sets, size_transitive_closure_node_pairs


def estimation_function_test(network, N):
    n_nodes = network.GetNodes()
    n_edges = network.GetEdges()
    n_z_nodes = network.CntDegNodes(0)
    max_degree = snap.MxDegree(network)
    
    rnd = snap.TRnd()
    rnd.Randomize()
    src_nodes = set()
    while len(src_nodes) < N:
        src_nodes.add(network.GetRndNId(rnd))
    
    size_tc = size_transitive_closure_node_ids(network, src_nodes)
    estimate = estimation_function(network, s=len(src_nodes))
    
    # rnd.Randomize()
    # dst_nodes = set()
    # while len(dst_nodes) < N:
    #     dst_nodes.add(network.GetRndNId(rnd))
    # size_tc = test_util.size_transitive_closure_node_pairs(network, node_id_pairs)
    # size_tc = size_transitive_closure_node_sets(network, src_nodes, dst_nodes)
    # estimate = estimation_function(network, s=len(src_nodes), t=len(dst_nodes))

    result = {'No. roots': N, 'No. nodes': n_nodes, 'No. edges': n_edges,
              'Max degree': max_degree, 'Zero degree': n_z_nodes,
              'TC size': size_tc, 'Estimate': estimate}

    return result


def estimation_function_paper_test(n_iters, N=1000, sizes=None, seed=None):
    df = pd.DataFrame(columns=['Iteration', 'Size', 'Graph type', 'No. roots', 
                               'No. nodes', 'No. edges', 'Max degree', 
                               'Zero degree', 'TC size', 'Estimate'])
    if sizes is None:
        sizes = [(100, 100), (100, 200), (100, 300), (100, 400), (100, 500),
                 (1000, 2000), (10000, 20000), (100000, 200000), 
                 (1000000, 2000000)]
    for (n_nodes, n_edges) in sizes:
        test_print(f'No. nodes: {n_nodes}; No. edges: {n_edges}')
        size_label = f'{n_nodes} + {n_edges}'
        for iteration in range(1, n_iters+1):
            test_print(f'\tIteration {iteration}')
            if seed is None:
                rnd = snap.TRnd()
            else:
                rnd = snap.TRnd(seed)
            network_rnd = snap.GenRndGnm(
                snap.TNEANet, n_nodes, n_edges, True, rnd)
            result_rnd = estimation_function_test(network_rnd, N)
            result_rnd['Graph type'] = 'Random'
            result_rnd['Iteration'] = iteration
            result_rnd['Size'] = size_label
            df = df.append(result_rnd, ignore_index=True)

            m = int(n_edges/n_nodes)
            # g = nx.barabasi_albert_graph(n_nodes, m)
            g = nx.extended_barabasi_albert_graph(n_nodes, m, 0.001, 0.81)
            network_sf = snap.TUNGraph.New()
            for n in g.nodes():
                network_sf.AddNode(n)
            for (s,d) in g.edges():
                network_sf.AddEdge(s, d)
            result_sf = estimation_function_test(network_sf, N)
            result_sf['Graph type'] = 'Scale-free'
            result_sf['Iteration'] = iteration
            result_sf['Size'] = size_label
            df = df.append(result_sf, ignore_index=True)
    
    results_filename = f'results/{current_date_time_str()}_estimation_function_paper_test_results.csv'
    df.to_csv(results_filename, index=False)
    
    return df


def estimation_function_real_data_test(n_iters, N=100, max_files_tested=inf, files_to_skip=[]):
    df = pd.DataFrame(columns=['Iteration', 'Dataset', 'Graph type', 
                               'No. roots', 'No. nodes', 'No. edges', 
                               'Max degree', 'Zero degree', 'TC size', 'Estimate'])
    # Get the data files and their sizes in the data directory
    files_roots_sizes = {}
    for root, dirs, files in os.walk("data/unlabeled"):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                size = os.stat(path).st_size
                files_roots_sizes[file] = (root, size)

    file_counter = 0
    for file, (root, size) in sorted(files_roots_sizes.items(), key=lambda item: item[1]):
        if file in files_to_skip:
            continue
        filepath = os.path.join(root, file)
        test_print(f'Loading dataset {file}')
        network = load_data.load_unlabeled_edge_file(filepath)
        n_nodes = network.GetNodes()
        n_edges = network.GetEdges()
        test_print(f'No. nodes: {n_nodes}; No. edges: {n_edges}')
        for iteration in range(1, n_iters+1):
            test_print(f'\tIteration {iteration}')
            result = estimation_function_test(network, N)
            result['Dataset'] = file
            result['Iteration'] = iteration
            df = df.append(result, ignore_index=True)
        
        file_counter += 1
        if file_counter == max_files_tested:
            break
            
    results_filename = f'results/{current_date_time_str()}_estimation_function_real_data_test_results.csv'
    df.to_csv(results_filename, index=False)
    
    return df
            
