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
    network = load_data.make_pg_paper_network()
    # network = make_gmark_network(GMarkUseCase.uniprot, size=25000)
    print(f"--> {elapsed_time(start_time)} Dataset loaded.")
    print(f"\t+++ Number of Nodes: {network.GetNodes()}")
    print(f"\t+++ Number of Edges: {network.GetEdges()}")
    summary = GraphMergeSummary(network)
    print(f"--> {elapsed_time(start_time)} Building evaluation graph...")
    summary.build_evalutation_graph()
    print(f"--> {elapsed_time(start_time)} Evaluation completed.")
    print(f"\t+++ Number of super nodes: {summary.evaluation_network.GetNodes()}")
    print(f"\t+++ Number of super edges: {summary.evaluation_network.GetEdges()}")
    # print(f"--> {elapsed_time(start_time)} Preparing for merging...")
    # merge = Merge(session, evaluation)
    # print(f"--> {elapsed_time(start_time)} Merging...")
    # merge.merge()
    # print(f"--> {elapsed_time(start_time)} Merging completed.")
    # print(f"\t+++ Number of hyper nodes: {merge.merge_graph.GetNodes()}")
    # print(f"\t+++ Number of hyper edges: {merge.merge_graph.GetEdges()}")
