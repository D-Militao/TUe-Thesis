from enum import Enum
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
from .test_util import TestTracker, elapsed_time, elapsed_time_str, stopwatch, current_date_time_str


class FullTest:
    class ResultsLabels(str, Enum):
        NETWORK = 'Network'
        N_NODES = 'No. nodes'
        N_EDGES = 'No. edges'
        N_Z_DEG = 'No. zero degree'
        MAX_DEG = 'Max degree'
        LOAD_TIME = 'Loading time'
        LOAD_MEM = 'Loading memory'
        LOAD_MEM_PEAK = 'Loading memory peak'
        TC_TIME = "TC time"
        TC_MEM = "TC memory"
        TC_MEM_PEAK = "TC memory peak"
        SKETCH_CREATION_TIME = "Sketch creation time"
        SKETCH_CREATION_MEM = "Sketch creation memory"
        SKETCH_CREATION_MEM_PEAK = "Sketch creation memory peak"
        SKETCH_EST_TIME = "Sketch estimation time"
        SKETCH_EST_MEM = "Sketch estimation memory"
        SKETCH_EST_MEM_PEAK = "Sketch estimation memory peak"

        def __str__(self) -> str:
            return str.__str__(self)

    def __init__(self, seed=42, N=1000, k_values=[5, 10, 50, 100], track_mem=False):
        self.seed = seed
        self.N = N
        self.k_values = k_values
        self.track_mem = track_mem
        self.results = {}
        for label in self.ResultsLabels:
            if label.startswith("Sketch"):
                for k in k_values:
                    self.results[label + f' k={k}'] = []
            else:
                self.results[label] = []
        self.test_tracker = TestTracker(track_mem)
        self.test_timestamp = current_date_time_str()

    def load_network(self, root, file):
        filepath = os.path.join(root, file)
        print(f"[{stopwatch()}] Loading {filepath}...")
        
        self.test_tracker.start()
        network = load_data.load_unlabeled_edge_file(filepath)
        load_time, load_mem, load_mem_peak = self.test_tracker.track()
        
        n_nodes = network.GetNodes()
        n_edges = network.GetEdges()
        n_z_deg = network.CntDegNodes(0)
        max_degree = snap.MxDegree(network)
        print(f"[{stopwatch()}] Nodes: {n_nodes}, Edges: {n_edges}")

        self.results[self.ResultsLabels.NETWORK].append(file)
        self.results[self.ResultsLabels.N_NODES].append(n_nodes)
        self.results[self.ResultsLabels.N_EDGES].append(n_edges)
        self.results[self.ResultsLabels.N_Z_DEG].append(n_z_deg)
        self.results[self.ResultsLabels.MAX_DEG].append(max_degree)
        self.results[self.ResultsLabels.LOAD_TIME].append(load_time)
        self.results[self.ResultsLabels.LOAD_MEM].append(load_mem)
        self.results[self.ResultsLabels.LOAD_MEM_PEAK].append(load_mem_peak)

        return network

    def test_all_distance_sketch_node_ids(self, network, node_ids, k):
        # Create the sketch for the given k
        print(f'[{stopwatch()}] Creating k={k} sketch...')
        self.test_tracker.start()
        graph_sketch = GraphSketch(network, k)
        graph_sketch.calculate_graph_sketch()
        sketch_time, sketch_mem, sketch_mem_peak = self.test_tracker.track()
        print(f'[{stopwatch()}] Finished creating k={k} sketch.')

        # Perform estimates on the sketch
        print(f'[{stopwatch()}] Estimating for N={node_ids.Len()} with k={k} sketch...')
        self.test_tracker.start()
        estimates = []  # snap.TFltV()
        for node_id in node_ids:
            estimates.append(graph_sketch.cardinality_estimation_node_id(node_id))
        est_time, est_mem, est_mem_peak = self.test_tracker.track()
        print(f'[{stopwatch()}] Finished estimating for N={node_ids.Len()} with k={k} sketch.')

        k_label = f' k={k}'
        self.results[self.ResultsLabels.SKETCH_CREATION_TIME+k_label].append(sketch_time)
        self.results[self.ResultsLabels.SKETCH_CREATION_MEM+k_label].append(sketch_mem)
        self.results[self.ResultsLabels.SKETCH_CREATION_MEM_PEAK+k_label].append(sketch_mem_peak)
        self.results[self.ResultsLabels.SKETCH_EST_TIME+k_label].append(est_time)
        self.results[self.ResultsLabels.SKETCH_EST_MEM+k_label].append(est_mem)
        self.results[self.ResultsLabels.SKETCH_EST_MEM_PEAK+k_label].append(est_mem_peak)

        return estimates

    def full_test_network(self, network) -> pd.DataFrame:
        node_results = pd.DataFrame()
        node_ids = snap.TIntV()

        rnd = snap.TRnd(self.seed)
        # Omit rnd.Randomize() line to get the same return values for different
        # program executions
        # rnd.Randomize()
        for i in range(self.N):
            node_ids.append(network.GetRndNId(rnd))
        node_results['Node ids'] = node_ids

        print(f'[{stopwatch()}] Calculating TC for N={node_ids.Len()} nodes...')
        tc_values = snap.TIntV()

        self.test_tracker.start()
        for node_id in node_ids:
            bfs_tree = network.GetBfsTree(node_id, True, False)
            tc_values.append(bfs_tree.GetNodes())
        tc_time, tc_mem, tc_mem_peak = self.test_tracker.track()

        print(f'[{stopwatch()}] Finished calculating TC for N={node_ids.Len()} nodes.')
        node_results['TC'] = tc_values
        self.results[self.ResultsLabels.TC_TIME].append(tc_time)
        self.results[self.ResultsLabels.TC_MEM].append(tc_mem)
        self.results[self.ResultsLabels.TC_MEM_PEAK].append(tc_mem_peak)

        for k in self.k_values:
            estimates = self.test_all_distance_sketch_node_ids(network, node_ids, k)
            node_results[f'Sketch k={k}'] = estimates

        # full_test_network_summary(network, node_ids, general_data, tc_data)
        return node_results


    def start_full_test(self):
        # Get the data files and their sizes in the data directory
        files_sizes = {}
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith("wiki-Vote.txt"):
                    path = os.path.join(root, file)
                    size = os.stat(path).st_size
                    files_sizes[file] = size

        for file, size in sorted(files_sizes.items(), key=lambda item: item[1]):
            network = self.load_network(root, file)

            node_results_df = self.full_test_network(network)
            node_results_filename = f"results/{self.test_timestamp}_{file}_node_results_N={self.N}.csv"
            node_results_df.to_csv(node_results_filename, index=False)

        # if not tracking memory, then remove memory columns
        results_df = pd.DataFrame.from_dict(self.results)
        if not self.track_mem:
            results_df = results_df[results_df.columns.drop(list(results_df.filter(regex='memory')))]
            results_filename = f"results/{self.test_timestamp}_results_N={self.N}.csv"
        else:
            results_filename = f"results/{self.test_timestamp}_memory_results_N={self.N}.csv"
        results_df.to_csv(results_filename, index=False)


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
