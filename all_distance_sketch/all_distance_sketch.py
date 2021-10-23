import random
import math
from pprint import pprint
from dataclasses import dataclass
import pickle
import json

import snap

from snap_util import snap_util


class GraphSketch:
    def __init__(self, graph, k, seed=None):
        self.graph = graph
        if isinstance(graph, snap.TUNGraph):
            self.tranposed_graph = graph
        else:
            self.tranposed_graph = snap_util.transpose_graph(graph)
        self.k = k
        self.node_ids = snap.TIntV()
        self.rankings = snap.TIntFltH()
        self.node_sketches = snap.TIntIntPrVH()

        if not seed == None:
            random.seed(seed)
        for NI in graph.Nodes():
            node_id = NI.GetId()
            self.node_ids.append(node_id)
            self.node_sketches[node_id] = snap.TIntPrV()
            
            rank = random.uniform(0, 1)
            self.rankings[node_id] = rank            

    def calculate_graph_sketch(self):
        for rankee_node_id, rankee_rank in sorted(self.rankings.items(), key=lambda item: item[1]):
            queue = snap.TIntV()
            queue.append(rankee_node_id)
            distances = snap.TIntH()
            distances[rankee_node_id] = 0
            # visited_nodes = set()
            # visited_nodes.add(rankee_node_id)
            visited_nodes = snap.TIntSet()
            visited_nodes.AddKey(rankee_node_id)

            while not queue.Empty():
                parent_node_id = queue.pop(0)
                parent_distance = distances[parent_node_id]

                # Do the insertion checks
                insert = False
                sketch = self.node_sketches[parent_node_id]
                sketch_len = sketch.Len()
                # if the node sketch isn't size k then we can always insert
                if sketch_len < self.k:
                    insert = True
                # remember, entries already in the node sketch all have a lower rank than the node we are currently considering
                # if the kth smallest distant in the node sketch is bigger then we can insert
                else:
                    k_smallest_dist = sketch[sketch_len - self.k].GetVal1()
                    if k_smallest_dist > parent_distance:
                        insert = True
                    elif k_smallest_dist == parent_distance and (sketch_len - self.k - 1) >= 0:
                        k_min_one_smallest_dist = (
                            sketch[sketch_len - self.k - 1].GetVal1())
                        if sketch_len == self.k:
                            insert = True
                        elif k_min_one_smallest_dist > parent_distance:
                            insert = True

                # We only continue the search if we insert
                if insert:
                    pair = snap.TIntPr(parent_distance, rankee_node_id)
                    self.node_sketches[parent_node_id].AddSorted(pair, False)

                    # continue bfs
                    NI = self.tranposed_graph.GetNI(parent_node_id)
                    out_node_ids = NI.GetOutEdges()
                    for out_node_id in out_node_ids:
                        if out_node_id not in visited_nodes:
                            visited_nodes.AddKey(out_node_id)
                            distances[out_node_id] = parent_distance + 1
                            queue.append(out_node_id)

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

    def save(self, filename):
        sketch_bin = filename+f'_k={self.k}_sketch.bin'
        ranking_bin = filename+f'_k={self.k}_ranking.bin'
        
        self.node_sketches.Save(snap.TFOut(sketch_bin))
        self.rankings.Save(snap.TFOut(ranking_bin))
    
    def load(self, filename):
        try:
            sketch_bin = filename+f'_k={self.k}_sketch.bin'
            ranking_bin = filename+f'_k={self.k}_ranking.bin'
            
            self.node_sketches.Load(snap.TFIn(sketch_bin))
            self.rankings.Load(snap.TFIn(ranking_bin))
        except RuntimeError as err:
            print(err)

    # ! ------------------------------------------------------------- ! #
    # ! Functions below are no longer in use and will be deleted soon ! #
    # !      They are temporarily being used for reference only       ! #
    # ! ------------------------------------------------------------- ! #

    def calculate_graph_sketch_slow(self):
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
