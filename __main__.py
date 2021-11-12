from enum import Enum
import random
import tracemalloc
import pickle

import snap 

from tests.full_test import FullTestUnlabeled, FullTestLabeled
from snap_util import load_data
from graph_merge_summary import GraphMergeSummary, Constants
from all_distance_sketch import GraphSketch, LabeledGraphSketch

def notebook_summary_test():
    network = snap.TNEANet.New()
    for i in range(1, 11):
        network.AddNode(i)

    # Component 1
    network.AddEdge(1, 2)
    network.AddEdge(2, 3)
    network.AddEdge(3, 1)
    
    # Component 2
    network.AddEdge(4, 5)
    network.AddEdge(5, 4)

    # Component 3
    network.AddEdge(6, 7)
    network.AddEdge(7, 6)
    network.AddEdge(6, 8)
    network.AddEdge(8, 7)

    # Other edges
    network.AddEdge(9, 3)
    network.AddEdge(2, 4)
    network.AddEdge(3, 6)
    network.AddEdge(5, 10)
    network.AddEdge(8, 10)

    labels = {}
    for NI in network.Nodes():
        labels[NI.GetId()] = str(NI.GetId())
    network.DrawGViz(snap.gvlDot, "results/1_network.png", " ", labels)

    summary = GraphMergeSummary(network, is_labeled=False)
    
    summary.build_evalutation_network()
    eval_labels = {}
    for NI in summary.evaluation_network.Nodes():
        eval_labels[NI.GetId()] = str(NI.GetId())
    summary.evaluation_network.DrawGViz(snap.gvlDot, "results/2_eval.png", " ", eval_labels)

    summary.build_merge_network(False)
    merge_labels = {}
    for NI in summary.merge_network.Nodes():
        merge_labels[NI.GetId()] = str(NI.GetId())
    summary.merge_network.DrawGViz(snap.gvlDot, "results/3_merge.png", " ", merge_labels)

    node_ids = [9, 5,10]
    for node_id in node_ids:
        est = summary.cardinality_estimation_labeled_node_id(node_id)
        bfs_tree = network.GetBfsTree(node_id, True, False)
        tc = bfs_tree.GetNodes()
        print(f'Node_ id: {node_id}; TC={tc}; Est={est}')

    for NI in network.Nodes():
        node_id = NI.GetId()
        super_node_id = summary.network.GetIntAttrDatN(node_id, Constants.META_NODE_ID)
        hyper_node_id = summary.evaluation_network.GetIntAttrDatN(super_node_id, Constants.META_NODE_ID)
        print(node_id, super_node_id, hyper_node_id)

def bfs(graph, root_node_id, labels):
    queue = snap.TIntV()
    queue.append(root_node_id)
    visited_nodes = snap.TIntSet()
    visited_nodes.AddKey(root_node_id)

    while not queue.Empty():
        src_node_id = queue.pop(0)
        NI = graph.GetNI(src_node_id)
        dst_node_ids = NI.GetOutEdges()
        for dst_node_id in dst_node_ids:
            if dst_node_id not in visited_nodes:
                edge_id = graph.GetEI(src_node_id, dst_node_id).GetId()
                label = graph.GetStrAttrDatE(edge_id, 'EDGE_LABEL')
                if label in labels:
                    visited_nodes.AddKey(dst_node_id)
                    queue.append(dst_node_id)
    return visited_nodes.Len()

def test_labeled_ads():
    graph = load_data.load_gmark_network(load_data.GMarkUseCase.shop, size=25000)
    labels = set()
    for EI in graph.Edges():
        edge_id = EI.GetId()
        label = graph.GetStrAttrDatE(edge_id, 'EDGE_LABEL')
        labels.add(label)
    labeled_graph_sketch = LabeledGraphSketch(graph, list(labels), 2, seed=42)
    labeled_graph_sketch.calculate_graph_sketch()
    graph_sketch = GraphSketch(graph, 2, seed=42)
    graph_sketch.calculate_graph_sketch()
    
    node_ids = snap.TIntV()
    rnd = snap.TRnd(42)
    # Omit rnd.Randomize() line to get the same return values for different
    # program executions
    # rnd.Randomize()
    for i in range(100):
        node_ids.append(graph.GetRndNId(rnd))
    
    random.seed(42)
    k_labels = int(len(labels)/2)
    rand_labels = random.choices(list(labels), k=k_labels)
    rand_labels = list(labels)
    tc_values = {}
    est_labeled_values = {}
    est_unlabeled_values = {}
    for node_id in node_ids:
        tc = bfs(graph, node_id, rand_labels)
        tc_values[node_id] = tc
        est_labeled = labeled_graph_sketch.cardinality_estimation_labels_node_id(node_id, rand_labels)
        est_labeled_values[node_id] = est_labeled
        est_unlabeled = graph_sketch.cardinality_estimation_node_id(node_id)
        est_unlabeled_values[node_id] = est_unlabeled
        print(f'Node id: {node_id}; TC: {tc}; Est unlabeled: {est_unlabeled}; Est labeled: {est_labeled}')
        # print(f'Node id: {node_id}; TC: {tc}; Est labeled: {est_labeled}')
        
        
if __name__ == '__main__':
    full_test_unlabeled = FullTestUnlabeled(
        calc_tc=True, test_sketch=True, test_summary=False, test_func=False, 
        seed=42, N=1000, k_values=[5, 10, 50], track_mem=False)
    full_test_unlabeled.start_full_test()
    
    # full_test_labeled = FullTestLabeled(
    #     calc_tc=True, test_sketch=True, test_summary=False, 
    #     seed=42, N=100, k_values=[5, 10, 50], track_mem=False)
    # full_test_labeled.start_full_test()
