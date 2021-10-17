import math
import pandas as pd

import snap


def estimation_function(network, s=None, t=None):
    n = network.GetNodes()
    m = network.GetEdges()
    if s == None:
        s = n
    if t == None:
        t = n

    saturation = (-(m-n)/n)
    estimation = s * t * (1 - math.exp(saturation))
    return estimation


def size_transitive_closure(network):
    size = 0
    for NI in network.Nodes():
        node_id = NI.GetId()
        bfs_tree = network.GetBfsTree(node_id, True, False)
        # NOTE We are counting a nodes connection to itself even if no such edge exists
        size += bfs_tree.GetNodes()
    return size


def size_transitive_closure_node_ids(network, node_ids):
    size = 0
    for node_id in node_ids:
        bfs_tree = network.GetBfsTree(node_id, True, False)
        # NOTE We are counting a nodes connection to itself even if no such edge exists
        size += bfs_tree.GetNodes()
    return size


def estimation_function_test_paper(seed, N=100):
    df = pd.DataFrame(columns=['Graph Type', 'No. roots', 'No. nodes',
                      'No. Edges', 'Max degree', 'Zero degree', 'TC Size', 
                      'Estimate'])
    sizes = [(100, 200), (1000, 2000), (10000, 20000),
             (100000, 200000), (1000000, 2000000)]
    rnd = snap.TRnd(seed)

    for (n_nodes, n_edges) in sizes:
        print(f'No. nodes: {n_nodes}; No. edges: {n_edges}')
        network_rnd = snap.GenRndGnm(
            snap.TNEANet, n_nodes, n_edges, True, rnd)
        result_rnd = estimation_function_test(network_rnd, seed, N)
        df = df.append(result_rnd, ignore_index=True)

        # network_sf = snap.GenRndGnm(
        #     snap.TNEANet, n_nodes, n_edges, True, rnd)
        # result_sf = estimation_function_test(network_sf, seed, N)
        # df = df.append(result_sf, ignore_index=True)
        
    return df


def estimation_function_test(network, seed, N=100):
    n_nodes = network.GetNodes()
    n_edges = network.GetEdges()
    n_z_nodes = network.CntDegNodes(0)
    max_degree = snap.MxDegree(network)

    node_ids = set()
    rnd = snap.TRnd(seed)
    while len(node_ids) < N:
        node_ids.add(network.GetRndNId(rnd))
    
    size_tc = size_transitive_closure_node_ids(network, node_ids)
    estimate = estimation_function(network, s=N)

    result = {'Graph Type': 'Random', 'No. roots': N, 
                'No. nodes': n_nodes, 'No. Edges': n_edges,
                'Max degree': max_degree, 'Zero degree': n_z_nodes, 
                'TC Size': size_tc, 'Estimate': estimate}
    
    return result