import random
import math
from pprint import pprint
from dataclasses import dataclass

import snap

import util


def least_descendant_subroutine(node_ids, graph):
    least_descendant_mapping = {}
    for node_id in node_ids:
        # Perform Dijkstra from the rankee node (node_id) on the transposed 
        # graph (False for traversing out edges, True for traversing in edges)
        bfs_tree = graph.GetBfsTree(node_id, False, True)
        for NI in bfs_tree.Nodes():
            bfs_node_id = NI.GetId()
            least_descendant_mapping[bfs_node_id] = node_id
            node_ids.remove(bfs_node_id)
            graph.DelNode(bfs_node_id)
    return least_descendant_mapping


def k_mins(graph, k):
    node_ids = snap.TIntV()
    for NI in graph.Nodes():
        node_id = NI.GetId()
        node_ids.append(node_id)

    for i in range(k):
        random.shuffle(node_ids)
        x = node_ids.copy()
        y = graph.ConvertGraph(snap.TNGraph)
        least_descendant_mapping = least_descendant_subroutine(x, y)
        pprint(least_descendant_mapping)


@dataclass
class NodeIdDistance:
    node_id: int
    distance: float    


class GraphSketch:
    def __init__(self, graph, k):
        self.graph = graph
        self.tranposed_graph = util.transpose_graph(graph)
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
            # node_id_dist is ordered by distance
            len_short_path, node_id_dist = self.tranposed_graph.GetShortPathAll(
                rankee_node_id, True)
            for dijkstra_node_id, dijkstra_dist in node_id_dist.items():
                insert = False
                sketch_len = self.node_sketches[dijkstra_node_id].Len()
                # if the node sketch isn't size k then we can always insert
                if sketch_len < self.k:
                    insert = True
                # remember, entries already in the node sketch all have a lower rank than the node we are currently considering
                # if the kth smallest distant in the node sketch is bigger then we can insert
                else:
                    k_smallest_dist = self.node_sketches[dijkstra_node_id][sketch_len - self.k].GetVal1()
                    
                    if k_smallest_dist  > dijkstra_dist:
                        insert = True
                    elif k_smallest_dist == dijkstra_dist and (sketch_len - self.k - 1) >= 0:
                        k_min_one_smallest_dist = self.node_sketches[dijkstra_node_id][sketch_len - self.k - 1].GetVal1()
                        if sketch_len == self.k:
                            insert = True
                        elif k_min_one_smallest_dist > dijkstra_dist:
                            insert = True
                
                if insert:
                    pair = snap.TIntPr(dijkstra_dist, rankee_node_id)
                    self.node_sketches[dijkstra_node_id].AddSorted(pair, False)

    def calculate_neighborhoods(self):
        for node_id, sketch in self.node_sketches.items():
            neighborhood = snap.TIntFltVH()
            for pair in sketch:
                dist = pair.GetVal1()
                pair_node_id = pair.GetVal2()
                ranking = self.rankings[pair_node_id]
                neighborhood.setdefault(dist, snap.TFltV()).AddSorted(ranking, True)

            self.neighborhoods[node_id] = neighborhood

    def estimate_cardinality(self, node_id, query_dist=math.inf):
        neighborhood = self.neighborhoods[node_id]

        all_neighbors = snap.TFltV()
        for dist, neighbors in neighborhood.items():
            if dist > query_dist:
                break
            all_neighbors.AddVMerged(neighbors)

        neighborhood_size = all_neighbors.Len()
        if neighborhood_size >= self.k:
            tau = all_neighbors[self.k]
            size_estimate = (self.k - 1) / tau
            print(neighborhood_size, self.k, tau, size_estimate)
        else:
            size_estimate = neighborhood_size

        return size_estimate