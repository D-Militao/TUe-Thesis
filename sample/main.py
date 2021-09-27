import time

from constants import Constants
from graph_merge_summary import GraphMergeSummary
import util
import load_data


def elapsed_time(start_time):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))


if __name__ == '__main__':
    start_time = time.time()
    print(f"--> {elapsed_time(start_time)} Loading dataset...")
    # network = load_data.make_pg_paper_network()
    network = load_data.make_gmark_network(load_data.GMarkUseCase.uniprot, size=100000)
    print(f"--> {elapsed_time(start_time)} Dataset loaded.")
    print(f"\t+++ Number of Nodes: {network.GetNodes()}")
    print(f"\t+++ Number of Edges: {network.GetEdges()}")
    summary = GraphMergeSummary(network)
    print(f"--> {elapsed_time(start_time)} Building evaluation network...")
    summary.build_evalutation_network()
    print(f"--> {elapsed_time(start_time)} Evaluation completed.")
    print(f"\t+++ Number of super nodes: {summary.evaluation_network.GetNodes()}")
    print(f"\t+++ Number of super edges: {summary.evaluation_network.GetEdges()}")
    print(f"--> {elapsed_time(start_time)} Building merge network...")
    summary.build_merge_network(is_target_merge=False)
    print(f"--> {elapsed_time(start_time)} Merging completed.")
    print(f"\t+++ Number of hyper nodes: {summary.merge_network.GetNodes()}")
    print(f"\t+++ Number of hyper edges: {summary.merge_network.GetEdges()}")
