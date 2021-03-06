from enum import Enum
from math import inf
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
from .test_util import TestTracker, test_print, current_date_time_str


class FullTest:
    class ResultsCol(str, Enum):
        # Network labels
        NETWORK = 'Network'
        N_NODES = 'No. nodes'
        N_EDGES = 'No. edges'
        N_Z_DEG = 'No. zero degree'
        MAX_DEG = 'Max degree'
        LOAD_TIME = 'Loading time'
        LOAD_MEM = 'Loading memory'
        LOAD_MEM_PEAK = 'Loading memory peak'
        # TC labels
        TC_TIME = "TC time"
        TC_MEM = "TC memory"
        TC_MEM_PEAK = "TC memory peak"
        # Sketch labels
        SKETCH_CREATION_TIME = "Sketch creation time"
        SKETCH_CREATION_MEM = "Sketch creation memory"
        SKETCH_CREATION_MEM_PEAK = "Sketch creation memory peak"
        SKETCH_EST_TIME = "Sketch estimation time"
        SKETCH_EST_MEM = "Sketch estimation memory"
        SKETCH_EST_MEM_PEAK = "Sketch estimation memory peak"
        # Summary labels
        SUMMARY_N_NODES = 'Summary no. nodes'
        SUMMARY_N_EDGES = 'Summary no. edges'
        SUMMARY_EVAL_TIME = "Summary evaluation time"
        SUMMARY_EVAL_MEM = "Summary evaluation memory"
        SUMMARY_EVAL_MEM_PEAK = "Summary evaluation memory peak"
        SUMMARY_MERGE_TIME = "Summary merge time"
        SUMMARY_MERGE_MEM = "Summary merge memory"
        SUMMARY_MERGE_MEM_PEAK = "Summary merge memory peak"
        SUMMARY_EST_TIME = "Summary estimation time"
        SUMMARY_EST_MEM = "Summary estimation memory"
        SUMMARY_EST_MEM_PEAK = "Summary estimation memory peak"
        # Estimation function labels
        # * there aren't any really, it's instant

        def __str__(self) -> str:
            return str.__str__(self)

    def __init__(self, test_sketch=True, test_summary=True, test_func=True, seed=42, N=1000, k_values=[5, 10, 50, 100], track_mem=False):
        self.test_sketch = test_sketch
        self.test_summary = test_summary
        self.test_func = test_func
        self.seed = seed
        self.N = N
        self.k_values = k_values
        self.track_mem = track_mem
        self.results = {}
        self.init_results()
        self.test_tracker = TestTracker(track_mem)
        self.test_timestamp = current_date_time_str()
        if not self.track_mem:
            self.results_filename = f"results/{self.test_timestamp}_results_N={self.N}.csv"
        else:
            self.results_filename = f"results/{self.test_timestamp}_memory_results_N={self.N}.csv"

    def init_results(self):
        self.results.clear()
        for label in self.ResultsCol:
            if label == self.ResultsCol.NETWORK:
                self.results[label] = snap.TStrV()
            elif label.startswith('Sketch'):
                if self.test_sketch:
                    for k in self.k_values:
                        self.results[label + f' k={k}'] = snap.TFltV()
            elif label.startswith('Summary'):
                if self.test_summary:
                    self.results[label] = snap.TFltV()
                    # if label.__contains__('evaluation'):
                    #     self.results[label] = snap.TFltV()
                    # else:
                    #     self.results[label+' is_target_merge=True'] = snap.TFltV()
                    #     self.results[label+' is_target_merge=False'] = snap.TFltV()
            else:
                self.results[label] = snap.TFltV()

    def load_network(self, root, file):
        filepath = os.path.join(root, file)
        test_print(f"+++++ Loading {filepath}...")
        
        self.test_tracker.start()
        network = load_data.load_unlabeled_edge_file(filepath)
        load_time, load_mem, load_mem_peak = self.test_tracker.track()
        
        n_nodes = network.GetNodes()
        n_edges = network.GetEdges()
        n_z_deg = network.CntDegNodes(0)
        max_degree = snap.MxDegree(network)
        test_print(f"+++++ Nodes: {n_nodes}, Edges: {n_edges}")

        self.results[self.ResultsCol.NETWORK].append(file)
        self.results[self.ResultsCol.N_NODES].append(n_nodes)
        self.results[self.ResultsCol.N_EDGES].append(n_edges)
        self.results[self.ResultsCol.N_Z_DEG].append(n_z_deg)
        self.results[self.ResultsCol.MAX_DEG].append(max_degree)
        self.results[self.ResultsCol.LOAD_TIME].append(load_time)
        self.results[self.ResultsCol.LOAD_MEM].append(load_mem)
        self.results[self.ResultsCol.LOAD_MEM_PEAK].append(load_mem_peak)

        return network

    def test_estimation_function(self, network):
        test_print(f"Estimating using the estimation function...")
        single_source_est = estimation_function(network, s=1)
        total_est = estimation_function(network, s=self.N)
        
        estimates = snap.TFltV()
        # TODO this is crap, find a better way
        for i in range(self.N):
            estimates.append(single_source_est)
        estimates.append(total_est)
        test_print(f"Finished estimating using the estimation function.")
        return estimates

    def test_graph_merge_summary(self, network, node_ids) -> dict:
        # merge_types = [False, True]
        
        # Create the summary
        test_print(f"Creating evaluation summary...")
        self.test_tracker.start()
        summary = GraphMergeSummary(network, is_labeled=False)
        summary.build_evalutation_network()
        eval_time, eval_mem, eval_mem_peak = self.test_tracker.track()
        summary_n_nodes = summary.evaluation_network.GetNodes()
        summary_n_edges = summary.evaluation_network.GetEdges()
        test_print(f"Finished creating evaluation summary.")
        
        self.results[self.ResultsCol.SUMMARY_EVAL_TIME].append(eval_time)
        self.results[self.ResultsCol.SUMMARY_EVAL_MEM].append(eval_mem)
        self.results[self.ResultsCol.SUMMARY_EVAL_MEM_PEAK].append(
            eval_mem_peak)
        
        # self.results[self.ResultsCol.SUMMARY_N_NODES].append(summary_n_nodes)
        # self.results[self.ResultsCol.SUMMARY_N_EDGES].append(summary_n_edges)
        
        # # Perform estimates on summary
        # test_print(f"Estimating for N={self.N} on summary...")
        # self.test_tracker.start()
        # estimates = snap.TFltV()
        # for node_id in node_ids:
        #     estimates.append(
        #         summary.cardinality_estimation_unlabeled_node_id(node_id))
        # estimates.append(sum(estimates))
        # est_time, est_mem, est_mem_peak = self.test_tracker.track()
        # test_print(f"Finished estimating for N={self.N} on summary.")

        # self.results[self.ResultsCol.SUMMARY_EST_TIME].append(est_time)
        # self.results[self.ResultsCol.SUMMARY_EST_MEM].append(est_mem)
        # self.results[self.ResultsCol.SUMMARY_EST_MEM_PEAK].append(est_mem_peak)

        # return estimates

        # Loop below is for using the merge graph
        test_print(f"Creating merge summary...")
        self.test_tracker.start()
        summary.build_merge_network()
        merge_time, merge_mem, merge_mem_peak = (self.test_tracker.track())
        summary_n_nodes = summary.merge_network.GetNodes()
        summary_n_edges = summary.merge_network.GetEdges()
        test_print(f"Finished creating merge summary.")

        # Perform estimates on summary
        test_print(f"Estimating for N={self.N} on merge summary...")
        self.test_tracker.start()
        estimates = snap.TFltV()
        for node_id in node_ids:
            estimates.append(summary.cardinality_estimation_node_id(node_id))
        estimates.append(sum(estimates))
        est_time, est_mem, est_mem_peak = self.test_tracker.track()
        test_print(f"Finished estimating for N={self.N} on merge summary.")

        # Add data to results
        self.results[self.ResultsCol.SUMMARY_N_NODES].append(summary_n_nodes)
        self.results[self.ResultsCol.SUMMARY_N_EDGES].append(summary_n_edges)
        self.results[self.ResultsCol.SUMMARY_MERGE_TIME].append(merge_time)
        self.results[self.ResultsCol.SUMMARY_MERGE_MEM].append(merge_mem)
        self.results[self.ResultsCol.SUMMARY_MERGE_MEM_PEAK].append(merge_mem_peak)
        self.results[self.ResultsCol.SUMMARY_EST_TIME].append(est_time)
        self.results[self.ResultsCol.SUMMARY_EST_MEM].append(est_mem)
        self.results[self.ResultsCol.SUMMARY_EST_MEM_PEAK].append(est_mem_peak)
        
        return estimates


    def test_all_distance_sketch(self, network, k, node_ids) -> snap.TFltV:
        # Create the sketch for the given k
        test_print(f"Creating k={k} sketch...")
        self.test_tracker.start()
        graph_sketch = GraphSketch(network, k, seed=self.seed)
        graph_sketch.calculate_graph_sketch()
        sketch_time, sketch_mem, sketch_mem_peak = self.test_tracker.track()
        test_print(f"Finished creating k={k} sketch.")

        # Perform estimates on the sketch
        test_print(f"Estimating for N={self.N} with k={k} sketch...")
        self.test_tracker.start()
        estimates = snap.TFltV()
        for node_id in node_ids:
            estimates.append(graph_sketch.cardinality_estimation_node_id(node_id))
        estimates.append(sum(estimates))
        est_time, est_mem, est_mem_peak = self.test_tracker.track()
        test_print(f"Finished estimating for N={self.N} with k={k} sketch.")

        # Add data to results
        k_label = f' k={k}'
        self.results[self.ResultsCol.SKETCH_CREATION_TIME+k_label].append(sketch_time)
        self.results[self.ResultsCol.SKETCH_CREATION_MEM+k_label].append(sketch_mem)
        self.results[self.ResultsCol.SKETCH_CREATION_MEM_PEAK+k_label].append(sketch_mem_peak)
        self.results[self.ResultsCol.SKETCH_EST_TIME+k_label].append(est_time)
        self.results[self.ResultsCol.SKETCH_EST_MEM+k_label].append(est_mem)
        self.results[self.ResultsCol.SKETCH_EST_MEM_PEAK+k_label].append(est_mem_peak)

        return estimates

    def full_test_network(self, network) -> pd.DataFrame:
        node_results = pd.DataFrame()
        node_ids = snap.TIntV()

        rnd = snap.TRnd(self.seed)
        # Omit rnd.Randomize() line to get the same return values for different
        # program executions
        rnd.Randomize()
        for i in range(self.N):
            node_ids.append(network.GetRndNId(rnd))
        node_ids.append(-1) # placeholder for total row
        node_results['Node ids'] = node_ids
        node_ids.pop(self.N)

        test_print(f"Calculating TC for N={self.N} nodes...")
        tc_values = snap.TIntV()
        self.test_tracker.start()
        for node_id in node_ids:
            bfs_tree = network.GetBfsTree(node_id, True, False)
            tc_values.append(bfs_tree.GetNodes())
        tc_values.append(sum(tc_values))
        tc_time, tc_mem, tc_mem_peak = self.test_tracker.track()

        test_print(f"Finished calculating TC for N={self.N} nodes.")
        node_results['TC'] = tc_values
        self.results[self.ResultsCol.TC_TIME].append(tc_time)
        self.results[self.ResultsCol.TC_MEM].append(tc_mem)
        self.results[self.ResultsCol.TC_MEM_PEAK].append(tc_mem_peak)

        if self.test_sketch:
            for k in self.k_values:
                sketch_estimates = self.test_all_distance_sketch(network, k, node_ids)
                node_results[f'Sketch k={k}'] = sketch_estimates
        
        if self.test_summary:
            summary_estimates = self.test_graph_merge_summary(network, node_ids)
            node_results[f'Summary'] = summary_estimates
        
        if self.test_func:
            func_estimates = self.test_estimation_function(network)
            node_results[f'Function'] = func_estimates

        return node_results


    def start_full_test(self, max_files_tested=inf):
        # Get the data files and their sizes in the data directory
        files_sizes = {}
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    size = os.stat(path).st_size
                    files_sizes[file] = size
                
        
        # create a file so that we can append after we have all the results of a network
        results_df = pd.DataFrame.from_dict(self.results)
        if not self.track_mem:
            results_df = results_df[results_df.columns.drop(list(results_df.filter(regex='memory')))]
        results_df.to_csv(self.results_filename, mode='a', index=False)

        file_counter = 0
        for file, size in sorted(files_sizes.items(), key=lambda item: item[1]):
            network = self.load_network(root, file)

            node_results_df = self.full_test_network(network)
            node_results_filename = f"results/{self.test_timestamp}_{file}_node_results_N={self.N}.csv"
            node_results_df.to_csv(node_results_filename, index=False)

            # if not tracking memory, then remove memory columns
            results_df = pd.DataFrame.from_dict(self.results)
            if not self.track_mem:
                results_df = results_df[results_df.columns.drop(list(results_df.filter(regex='memory')))]
            # append results to the file created earlier
            results_df.to_csv(self.results_filename, mode='a', header=False, index=False)
            # reinitialize the results dict
            self.init_results()

            file_counter += 1
            if file_counter == max_files_tested:
                break

# Used previously to test gmark and will be needed in the future
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
