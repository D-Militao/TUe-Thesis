import random
import math
from pprint import pprint
from dataclasses import dataclass

import snap

from snap_util import snap_util


class GraphSketch:
    def __init__(self, graph, k):
        self.graph = graph
        if isinstance(graph, snap.TUNGraph):
            self.tranposed_graph = graph
        else:
            self.tranposed_graph = snap_util.transpose_graph(graph)
        self.k = k
        self.node_ids = snap.TIntV()
        self.rankings = {}
        self.node_sketches = {}
        self.neighborhoods = {}

        for NI in graph.Nodes():
            node_id = NI.GetId()
            self.node_ids.append(node_id)
            self.node_sketches[node_id] = snap.TIntPrV()

        for node_id in self.node_ids:
            rank = random.uniform(0, 1)
            self.rankings[node_id] = rank

    def calculate_graph_sketch(self):
        for rankee_node_id, rankee_rank in sorted(self.rankings.items(), key=lambda item: item[1]):
            # node_id_dist is ordered by distance in ascending order
            len_short_path, node_id_shortest_dist = self.tranposed_graph.GetShortPathAll(
                rankee_node_id, True)
            for dijkstra_node_id, dijkstra_dist in node_id_shortest_dist.items():
                insert = False
                sketch = self.node_sketches[dijkstra_node_id]
                sketch_len = sketch.Len()
                # if the node sketch isn't size k then we can always insert
                if sketch_len < self.k:
                    insert = True
                # remember, entries already in the node sketch all have a lower rank than the node we are currently considering
                # if the kth smallest distant in the node sketch is bigger then we can insert
                else:
                    k_smallest_dist = sketch[sketch_len - self.k].GetVal1()
                    if k_smallest_dist > dijkstra_dist:
                        insert = True
                    elif k_smallest_dist == dijkstra_dist and (sketch_len - self.k - 1) >= 0:
                        k_min_one_smallest_dist = sketch[sketch_len -
                                                         self.k - 1].GetVal1()
                        if sketch_len == self.k:
                            insert = True
                        elif k_min_one_smallest_dist > dijkstra_dist:
                            insert = True

                if insert:
                    pair = snap.TIntPr(dijkstra_dist, rankee_node_id)
                    self.node_sketches[dijkstra_node_id].AddSorted(pair, False)

    def graph_sketch_bfs(self, rankee_node_id: int):
        distances = {}
        colors = {}
        parents = {}
        queue = snap.TIntV()
        queue.append(rankee_node_id)
        while not queue.Empty():
            pop_index = queue.Len() - 1
            node_id = queue.pop(pop_index)
            NI = self.tranposed_graph.GetNI(node_id)
            out_node_ids = NI.GetOutEdges()
            for out_node_id in out_node_ids:
                if out_node_id not in colors:
                    # ! do the insertion checks
                    colors[out_node_id] = 0
                    distances[out_node_id] = distances[node_id] + 1
                    parents[out_node_id] = node_id
                    queue.append(out_node_id)
            colors[node_id] = 1

        # NI = self.tranposed_graph.GetNI(dijkstra_node_id)
        # out_node_ids = NI.GetOutEdges()
        # for out_node_id in out_node_ids:
        #     insert = False
        #     sketch = self.node_sketches[out_node_id]
        #     sketch_len = sketch.Len()
        #     # if the node sketch isn't size k then we can always insert
        #     if sketch_len < self.k:
        #         insert = True
        #     # remember, entries already in the node sketch all have a lower rank than the node we are currently considering
        #     # if the kth smallest distant in the node sketch is bigger then we can insert
        #     else:
        #         k_smallest_dist = sketch[sketch_len - self.k].GetVal1()
        #         if k_smallest_dist > dist:
        #             insert = True
        #         elif k_smallest_dist == dist and (sketch_len - self.k - 1) >= 0:
        #             k_min_one_smallest_dist = sketch[sketch_len -
        #                                                 self.k - 1].GetVal1()
        #             if sketch_len == self.k:
        #                 insert = True
        #             elif k_min_one_smallest_dist > dist:
        #                 insert = True

        #     if insert:
        #         pair = snap.TIntPr(dist, rankee_node_id)
        #         self.node_sketches[out_node_id].AddSorted(pair, False)
        #         self.graph_sketch_bfs(rankee_node_id, )
        #     else:
        #         return

    def calculate_graph_sketch_fast(self):
        for rankee_node_id, rankee_rank in sorted(self.rankings.items(), key=lambda item: item[1]):
            # node_id_dist is ordered by distance in ascending order
            pair = snap.TIntPr(0, rankee_node_id)
            self.node_sketches[rankee_node_id].AddSorted(pair, False)
            self.graph_sketch_bfs(rankee_node_id, rankee_node_id, 1)
                
            len_short_path, node_id_shortest_dist = self.tranposed_graph.GetShortPathAll(
                rankee_node_id, True)
            for dijkstra_node_id, dijkstra_dist in node_id_shortest_dist.items():
                insert = False
                sketch = self.node_sketches[dijkstra_node_id]
                sketch_len = sketch.Len()
                # if the node sketch isn't size k then we can always insert
                if sketch_len < self.k:
                    insert = True
                # remember, entries already in the node sketch all have a lower rank than the node we are currently considering
                # if the kth smallest distant in the node sketch is bigger then we can insert
                else:
                    k_smallest_dist = sketch[sketch_len - self.k].GetVal1()
                    if k_smallest_dist > dijkstra_dist:
                        insert = True
                    elif k_smallest_dist == dijkstra_dist and (sketch_len - self.k - 1) >= 0:
                        k_min_one_smallest_dist = sketch[sketch_len -
                                                         self.k - 1].GetVal1()
                        if sketch_len == self.k:
                            insert = True
                        elif k_min_one_smallest_dist > dijkstra_dist:
                            insert = True

                if insert:
                    pair = snap.TIntPr(dijkstra_dist, rankee_node_id)
                    self.node_sketches[dijkstra_node_id].AddSorted(pair, False)

    def cardinality_estimation_node_id(self, node_id, dist=math.inf):
        sketch = self.node_sketches[node_id]
        neighbors = snap.TFltV()
        for pair in sketch:
            pair_dist = pair.GetVal1()
            if pair_dist <= dist:
                pair_node_id = pair.GetVal2()
                pair_ranking = self.rankings[pair_node_id]
                neighbors.AddSorted(pair_ranking, True)

        neighborhood_size = neighbors.Len()
        if neighborhood_size >= self.k:
            tau = neighbors[self.k - 1]
            size_estimate = (self.k - 1) / tau
        else:
            size_estimate = neighborhood_size

        return size_estimate

    def calculate_neighborhoods(self):
        for node_id, sketch in self.node_sketches.items():
            neighborhood = {}
            for pair in sketch:
                pair_dist = pair.GetVal1()
                pair_node_id = pair.GetVal2()
                pair_ranking = self.rankings[pair_node_id]
                neighborhood.setdefault(pair_dist, snap.TFltV()).AddSorted(
                    pair_ranking, True)
            self.neighborhoods[node_id] = neighborhood

    def cardinality_estimation_neighborhood(self, node_id, query_dist=math.inf):
        neighborhood = self.neighborhoods[node_id]
        all_neighbors = snap.TFltV()
        counter = 0
        for dist, neighbors in neighborhood.items():
            if dist > query_dist:
                break
            counter += neighbors.Len()
            all_neighbors.AddVMerged(neighbors)

        neighborhood_size = all_neighbors.Len()
        if neighborhood_size >= self.k:
            tau = all_neighbors[self.k - 1]
            size_estimate = (self.k - 1) / tau
        else:
            size_estimate = neighborhood_size

        return size_estimate
