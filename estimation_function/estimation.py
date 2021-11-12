import math
import pandas as pd
import networkx as nx

import snap


def estimation_function(network, s=None, t=None):
    n = network.GetNodes()
    m = network.GetEdges()
    if s is None:
        s = n
    if t is None:
        t = n

    saturation = (-(m - n) / n)
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

def size_transitive_closure_node_pairs(network, src_node_ids, dst_node_ids, N):
    size = 0
    for src_node_id in src_node_ids:
        for dst_node_id in dst_node_ids:
            # src_node_id = src_node_ids[idx]
            # dst_node_id = dst_node_ids[idx]
            bfs_tree = network.GetBfsTree(src_node_id, True, False)
            for NI in bfs_tree.Nodes():
                node_id = NI.GetId()
                if node_id == dst_node_id:
                    size += 1
                    break
    return size


def estimation_function_test_paper(N=1000, seed=None):
    df = pd.DataFrame(columns=['Iteration', 'Size', 'Graph type', 'No. roots', 'No. nodes',
                               'No. edges', 'Max degree', 'Zero degree', 'TC size',
                               'Estimate'])
    sizes = [(100, 100), (100, 200), (100, 300), (100, 400), (100, 500),
             (1000, 2000), (10000, 20000),(100000, 200000), (1000000, 2000000),]
    if seed is None:
        rnd = snap.TRnd(42)
    else:
        rnd = snap.TRnd(42)
    n_iter = 5
    for (n_nodes, n_edges) in sizes:
        print(f'No. nodes: {n_nodes}; No. edges: {n_edges}')
        for iteration in range(1, n_iter+1):
            network_rnd = snap.GenRndGnm(
                snap.TNEANet, n_nodes, n_edges, True, rnd)
            result_rnd = estimation_function_test(network_rnd, N, rnd)
            result_rnd['Graph type'] = 'Random'
            result_rnd['Iteration'] = iteration
            result_rnd['Size'] = str(n_nodes) + str(n_edges)
            df = df.append(result_rnd, ignore_index=True)

            m = int(n_edges/n_nodes)
            # g = nx.barabasi_albert_graph(n_nodes, m)
            g = nx.extended_barabasi_albert_graph(n_nodes, m, 0.001, 0.9)
            network_sf = snap.TUNGraph.New()
            for n in g.nodes():
                network_sf.AddNode(n)
            for (s,d) in g.edges():
                network_sf.AddEdge(s, d)
            # network_sf = snap.GenRndGnm(
            #     snap.TNEANet, n_nodes, n_edges, True, rnd)
            result_sf = estimation_function_test(network_sf, N, rnd)
            result_sf['Graph type'] = 'Scale-free'
            result_sf['Iteration'] = iteration
            result_sf['Size'] = str(n_nodes) + str(n_edges)
            df = df.append(result_sf, ignore_index=True)

    return df


def estimation_function_test(network, N, rnd):
    n_nodes = network.GetNodes()
    n_edges = network.GetEdges()
    n_z_nodes = network.CntDegNodes(0)
    max_degree = snap.MxDegree(network)
    
    node_ids = set()
    src_node_ids = set()
    dst_node_ids = set()
    while len(src_node_ids) < N:
        # node_ids.add(network.GetRndNId(rnd))
        src_node_ids.add(network.GetRndNId(rnd))
        dst_node_ids.add(network.GetRndNId(rnd))

    # size_tc = size_transitive_closure_node_ids(network, node_ids)
    size_tc = size_transitive_closure_node_pairs(network, src_node_ids, dst_node_ids, N)
    estimate = estimation_function(network, s=N, t=N)

    result = {'No. roots': N, 'No. nodes': n_nodes, 'No. edges': n_edges,
              'Max degree': max_degree, 'Zero degree': n_z_nodes,
              'TC size': size_tc, 'Estimate': estimate}

    return result
