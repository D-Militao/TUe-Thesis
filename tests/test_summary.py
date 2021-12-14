from functools import partial
import pandas as pd

import snap

from .context import snap_util, load_data
from .context import GraphMergeSummary
from .test_util import TestTracker, test_print, current_date_time_str


def graph_merge_summary_test(network) -> dict:
    result = {}
    test_tracker = TestTracker()
    merge_types = [False, True]
    
    # Create the summary
    test_print(f'Creating evaluation summary')
    test_tracker.start()
    summary = GraphMergeSummary(network, is_labeled=True)
    summary.build_evalutation_network()
    summary_time = test_tracker.track()
    summary_n_nodes = summary.evaluation_network.GetNodes()
    summary_n_edges = summary.evaluation_network.GetEdges()
    test_print(f"Finished creating evaluation summary.")
    
    result['Summary time'] = summary_time
    result['No. nodes summary'] = summary_n_nodes
    result['No. edges summary'] = summary_n_edges
    
    for is_target_merge in merge_types:
        test_print(f"Creating is_target_merge={is_target_merge} merge summary...")
        test_tracker.start()
        summary.build_merge_network(is_target_merge=is_target_merge)
        merge_time = test_tracker.track()
        merge_n_nodes = summary.merge_network.GetNodes()
        merge_n_edges = summary.merge_network.GetEdges()
        test_print(f"Finished creating is_target_merge={is_target_merge} merge summary.")
        
        if is_target_merge:
            result['Target merge time'] = merge_time
            result['No. nodes target merge'] = merge_n_nodes
            result['No. edges target merge'] = merge_n_edges
        else:
            result['Source merge time'] = merge_time
            result['No. nodes source merge'] = merge_n_nodes
            result['No. edges source merge'] = merge_n_edges
            
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

    # Perform estimates on summary
    # test_print(f"Estimating for N={self.N} on merge summary...")
    # self.test_tracker.start()
    # estimates = snap.TFltV()
    # for node_id in node_ids:
    #     estimates.append(summary.cardinality_estimation_labeled_node_id(node_id))
    # estimates.append(sum(estimates))
    # est_time, est_mem, est_mem_peak = self.test_tracker.track()
    # test_print(f"Finished estimating for N={self.N} on merge summary.")

    return result


def graph_merge_summary_test_gmark(n_iters=5):
    df = pd.DataFrame(columns=['Iteration', 'Dataset', 'No. nodes', 'No. edges', 
                               'No. nodes evaluation', 'No. edges evaluation', 'Evaluation time',
                               'No. nodes target merge', 'No. edges target merge', 'Target merge time',
                               'No. nodes source merge', 'No. edges source merge', 'Source merge time',
                               ]) # 'TC size', 'Estimate'
    network_names_methods = {
        # 'shop_1k': partial(load_data.load_gmark_network,load_data.GMarkUseCase.shop, size=1000),
        # 'shop_5k': partial(load_data.load_gmark_network,load_data.GMarkUseCase.shop, size=5000),
        # 'shop_25k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.shop, size=25000),
        # 'shop_50k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.shop, size=50000),
        # 'shop_100k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.shop, size=100000),
        # 'shop_200k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.shop, size=200000),
        
        # 'bib_1k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.test, size=1000),
        # 'bib_5k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.test, size=5000),
        # 'bib_25k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.test, size=25000),
        # 'bib_50k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.test, size=50000),
        # 'bib_100k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.test, size=100000),
        # 'bib_200k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.test, size=200000),
        
        # 'uni_1k': partial(load_data.load_gmark_network,load_data.GMarkUseCase.uniprot, size=1000),
        # 'uni_5k': partial(load_data.load_gmark_network,load_data.GMarkUseCase.uniprot, size=5000),
        # 'uni_25k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.uniprot, size=25000),
        # 'uni_50k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.uniprot, size=50000),
        # 'uni_100k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.uniprot, size=100000),
        'uni_200k': partial(load_data.load_gmark_network, load_data.GMarkUseCase.uniprot, size=200000),
    }
    
    for network_name, load_network_method in network_names_methods.items():
        test_print(f'Loading dataset {network_name}')
        network = load_network_method()
        n_nodes = network.GetNodes()
        n_edges = network.GetEdges()
        test_print(f'No. nodes: {n_nodes}; No. edges: {n_edges}')
        for iteration in range(1, n_iters+1):
            test_print(f'Iteration {iteration}')
            result = graph_merge_summary_test(network)
            result['No. nodes'] = n_nodes
            result['No. edges'] = n_edges
            result['Dataset'] = network_name
            result['Iteration'] = iteration
            df = df.append(result, ignore_index=True)
    
    results_filename = f'results/{current_date_time_str()}_summary_gmark_test_results.csv'
    df.to_csv(results_filename, index=False)
    
    return df