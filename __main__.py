from tests import full_test

if __name__ == '__main__':
    full_test.full_test(42, N=1000)
    

    # graph = load_data.load_unlabeled_edge_file("data/wiki-Vote.txt")
    # print(graph.GetNodes(), graph.GetEdges())

    # print(f"--> {elapsed_time_str(__start_time__)} Loading dataset...")
    # # network = load_data.make_pg_paper_network()
    # network = load_data.make_gmark_network(load_data.GMarkUseCase.shop, size=1000)
    # print(f"--> {elapsed_time_str(__start_time__)} Dataset loaded.")
    # print(f"\t+++ Number of Nodes: {network.GetNodes()}")
    # print(f"\t+++ Number of Edges: {network.GetEdges()}")

    # test_graph_merge_summary(network)

    # result = estimation.estimation_function_test(network, 42)
    # result = estimation.estimation_function_test_paper(42)
    # print(result)

    # test_ads(network, 30)