from enum import Enum
import tracemalloc
import pickle

from guppy import hpy
import snap 

from tests.full_test import FullTest
from snap_util import load_data
from graph_merge_summary import GraphMergeSummary, Constants
from all_distance_sketch import GraphSketch

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
        est = summary.cardinality_estimation_node_id(node_id)
        bfs_tree = network.GetBfsTree(node_id, True, False)
        tc = bfs_tree.GetNodes()
        print(f'Node_ id: {node_id}; TC={tc}; Est={est}')

    for NI in network.Nodes():
        node_id = NI.GetId()
        super_node_id = summary.network.GetIntAttrDatN(node_id, Constants.META_NODE_ID)
        hyper_node_id = summary.evaluation_network.GetIntAttrDatN(super_node_id, Constants.META_NODE_ID)
        print(node_id, super_node_id, hyper_node_id)

if __name__ == '__main__':
    full_test = FullTest(test_sketch=True, test_summary=True, test_func=True, 
                         seed=42, N=1000, k_values=[5, 10, 50, 100], track_mem=False)
    full_test.start_full_test()
    exit()
