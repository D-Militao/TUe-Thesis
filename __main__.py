from enum import Enum
import random
import pandas as pd

import snap 

from tests.full_test import FullTestUnlabeled, FullTestLabeled
from snap_util import load_data
from graph_merge_summary import GraphMergeSummary, UnlabeledGraphSummary, Constants
from all_distance_sketch import GraphSketch, LabeledGraphSketch

import estimation_function
import tests.test_estimation as test_estimation
import tests.test_summary as test_summary
import tests.plot_results as plot_results
from tests.test_util import current_date_time_str, current_time_str

def notebook_example_labeled_sketch_test():
    # network = snap.TNEANet.New()
    # for i in range(1, 9+1):
    #     network.AddNode(i)

    # # Component 1
    # network.AddEdge(2, 1)
    # network.AddEdge(2, 4)
    # network.AddEdge(3, 2)
    # network.AddEdge(3, 5)
    # network.AddEdge(4, 5)
    # network.AddEdge(5, 2)
    
    # # Component 2
    # # network.AddEdge(6, 3)
    # # network.AddEdge(7, 2)
    # # network.AddEdge(8, 1)
    # # network.AddEdge(9, 1)
    
    network = load_data.load_labeled_sketch_example()
    sketch = LabeledGraphSketch(network, 2)
    sketch.calculate_graph_sketch()
    print('##### Rankings #####')
    for node, rank in sketch.rankings.items():
        print(f'Node:{node}; Rank: {rank}')
        
    print('##### Sketches #####')
    for label, sketches in sketch.labels_node_sketches.items():
        print(f'Label: {label}')
        for node, n_sketch in sketches.items():
            print(f'\tADS({node}):')
            for pair in n_sketch:
                dist = pair.GetVal1()
                node_s = pair.GetVal2()
                print(f'\t\t{node_s}, {dist}')

def load_notebook_example_unlabeled():
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
    
    return network

def notebook_example_unlabeled_summary_test():
    network = load_notebook_example_unlabeled()

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

    
def run_estimation_function_paper_test():
    n_iters = 5
    
    # sizes = []
    # x = 5000
    # while x < 50001:
    #     sizes.append((5000, x))
    #     if x < 20000:
    #         x += 2500
    #     elif x < 30000:
    #         x += 5000
    #     else:
    #         x += 10000
    
    sizes = [(25, 50)]
    for i in range(5):
        for (x, y) in [(50, 100),(100, 200), (250, 500)]:
            sizes.append((x*10**i, y*10**i))
    print(sizes)        
    
    # sizes = [(100, 100), (100, 200), (100, 300), (100, 400), (100, 500), (100, 1000), (1000, 2000)]
    # sizes = [(100000, 200000), (1000000, 2000000)]
    # (10000, 20000), (100000, 200000), (1000000, 2000000)
    df = test_estimation.estimation_function_paper_test(n_iters, N=10, sizes=sizes)
    

def run_estimation_function_real_data_test():
    n_iters = 5
    files_to_skip = ['wiki-Vote.txt', 'wiki-Vote.txt', 'web-NotreDame.txt', 
                     'web-Stanford.txt', 'twitter_combined.txt', 
                     'soc-Epinions1.txt', 'amazon0601.txt', 'web-Google.txt', 
                     'web-BerkStan.txt', ]
    df = test_estimation.estimation_function_real_data_test(
        n_iters, N=10, max_files_tested=100, files_to_skip=files_to_skip)


def run_graph_merge_summary_test_gmark():
    n_iters = 1
    test_summary.graph_merge_summary_test_gmark(n_iters=n_iters)

if __name__ == '__main__':
    # filename = 'data/unlabeled/web-NotreDame.txt'
    # network = load_data.load_unlabeled_edge_file(filename)
    # # network = load_notebook_example_unlabeled()
    # summary = UnlabeledGraphSummary(network)
    # summary.build_summary()
    # print(summary.merge_network.GetNodes(), summary.merge_network.GetEdges())
    # print(summary.cardinality_estimation_node_id(155762))
    # notebook_example_labeled_sketch_test()
    # run_graph_merge_summary_test_gmark()
    # run_estimation_function_real_data_test()
    # run_estimation_function_paper_test()
    
    ##########
    # SKETCH #
    ##########
    # Plot ads paper results
    # filepath = 'results/final/sketch/ads_paper_results.csv'
    # plot_results.results_paper_plots(filepath)
    
    # Plot sketch unlabeled data results
    # folder_path = 'results/v3-results'
    # drop_headers = ['Function', 'Summary', 'Sketch k=100']
    # drop_headers.append('Sketch k=5')
    # drop_headers.append('Sketch k=10')
    # plot_results.sketch_results_plots(
    #     folder_path, drop_headers, 
    #     plot_datasets_separate=False, plot_sketches_separate=True, plot_combined=False)
    
    # Plot sketch times
    # filepath = 'results/final/sketch/ads_rw_times.csv'
    # plot_results.sketch_time_plots(filepath)
    
    ###########
    # SUMMARY #
    ###########
    # Plot summary gmark compression results
    # filepath = 'results/final/summary/summary_gmark_test_results_compression.csv'
    # plot_results.summary_plots_gmark_compression(filepath)
    
    # Plot summary gmark query results
    # filepath = 'results/final/summary/summary_gmark_test_results_queries.csv'
    # plot_results.summary_plots_gmark_queries(filepath)
    
    
    ##############
    # ESTIMATION #
    ##############
    # Plot est func paper 5k node graph results
    # filepath = 'results/final/estimation/estimation_function_paper_test_results_5k_nodes.csv'
    # plot_results.estimation_plot_paper(filepath, False)
    
    # # Plot est func paper density=2 graph 
    # filepath = 'results/final/estimation/estimation_function_paper_test_results_density_2.csv'
    # plot_results.estimation_plot_paper(filepath, True)
    
    # Plot est func real world datasets
    # filepath = 'results/final/estimation/estimation_function_real_world_results.csv'
    # plot_results.estimation_plot_real_world(filepath)
    
    # Plot est func real world datasets individual nodes
    # folder_path = 'results/final/estimation/individual_results'
    # plot_results.estimation_results_plot_rw(folder_path)
    
    ##################
    # LABELED SKETCH #
    ##################
    
    
    #####################
    # UNLABELED SUMMARY #
    #####################
    
    # Plot unlabeled summary compression and time
    # filepath = 'results/final/unlabeled_summary/2021-12-04_18h27m20s_results_N=100.csv'
    # filepath = 'results/2021-12-06_22h26m59s_results_N=100.csv'
    # plot_results.unlabeled_summary_plots(filepath)
    
    
    
    full_test_unlabeled = FullTestUnlabeled(
        calc_tc=True, test_sketch=False, test_summary=True, test_func=False, 
        N=100, k_values=[5, 10, 50]) # seed=42
    full_test_unlabeled.start_full_test(max_files_tested=100, num_files_skip=0)
    
    # full_test_labeled = FullTestLabeled(
    #     calc_tc=True, test_sketch=True, test_summary=False, 
    #     seed=42, N=100, k_values=[5, 10, 50])
    # full_test_labeled.start_full_test()



    # full_test_unlabeled = FullTestUnlabeled(
    #     calc_tc=True, test_sketch=True, test_summary=False, test_func=False, 
    #     seed=42, N=100, k_values=[5, 10, 50])
    # full_test_unlabeled.start_full_test()