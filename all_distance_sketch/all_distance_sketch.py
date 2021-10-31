import multiprocessing
import random
import math
import tqdm
from multiprocessing import Pool, TimeoutError
from pprint import pprint
from dataclasses import dataclass

import snap

from snap_util import snap_util
from snap_util import load_data


class GraphSketch:
    def __init__(self, graph, k: int, seed=None, rankings=None):
        self.graph = graph
        if isinstance(graph, snap.TUNGraph):
            self.tranposed_graph = graph
        else:
            self.tranposed_graph = snap_util.transpose_graph(graph)
        self.k = k
        self.node_ids = snap.TIntV()
        if rankings is None:
            self.rankings = snap.TIntFltH()
        else:
            self.rankings = rankings
        
        # self.node_sketches = snap.TIntIntPrVH()
        self.node_sketches = {}

        if seed is not None:
            random.seed(seed)
        for NI in graph.Nodes():
            node_id = NI.GetId()
            self.node_ids.append(node_id)
            self.node_sketches[node_id] = snap.TIntPrV()
            
            if rankings is None:
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
                    elif k_smallest_dist == parent_distance:
                        insert
                        if sketch_len == self.k:
                            insert = True
                        else:
                            k_min_one_smallest_dist = (
                                sketch[sketch_len - self.k - 1].GetVal1())
                            if k_min_one_smallest_dist != parent_distance:
                                insert = True

                # We only continue the search if we insert
                if insert:
                    pair = snap.TIntPr(parent_distance, rankee_node_id)
                    self.node_sketches[parent_node_id].AddSorted(pair, False) # descending order
                    # continue bfs
                    NI = self.tranposed_graph.GetNI(parent_node_id)
                    out_node_ids = NI.GetOutEdges()
                    for out_node_id in out_node_ids:
                        if out_node_id not in visited_nodes:
                            visited_nodes.AddKey(out_node_id)
                            distances[out_node_id] = parent_distance + 1
                            queue.append(out_node_id)
                            
    def print_sketch(self):
        for rankee_node_id, sketch in self.node_sketches.items():
            print(rankee_node_id)
            for pair in sketch:
                dist = pair.GetVal1()
                node_id = pair.GetVal2()
                print(f'\tNode id: {node_id}; Dist: {dist}')

    def cardinality_estimation_bottom_k_node_id(self, node_id, dist=math.inf):
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
    
    def cardinality_estimation_hip_node_id(self, node_id, dist=math.inf):
        sketch = self.node_sketches[node_id]
        tau_values = {}
        for i in range(sketch.Len()):
            tau_pair = sketch[i]
            tau_dist = tau_pair.GetVal1()
            if tau_dist <= dist:
                tau_node_id = tau_pair.GetVal2()
                tau_ranking = self.rankings[tau_node_id]
                tau_neighbors = snap.TFltV()
                
                if sketch.Len()-i < self.k:
                    tau_values[tau_node_id] = 1
                else:
                    for j in range(i, sketch.Len()):
                        pair = sketch[j]
                        pair_node_id = pair.GetVal2()
                        pair_ranking = self.rankings[pair_node_id]
                        tau_neighbors.AddSorted(pair_ranking, True)
                    
                    tau_values[tau_node_id] = tau_neighbors[self.k - 1]
        
        size_estimate = 0
        for tau_value in tau_values.values():
            a = (1 / tau_value)
            size_estimate += a
        
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


class LabeledGraphSketch:
    def __init__(self, graph, k: int, seed=None):
        self.graph = graph
        if isinstance(graph, snap.TUNGraph):
            self.tranposed_graph = graph
        else:
            self.tranposed_graph = snap_util.transpose_graph(graph)
        self.labels = snap_util.get_edges_attribute_values(graph, load_data.__edge_label__)
        self.k = k
        self.node_ids = snap.TIntV()
        self.rankings = snap.TIntFltH()
        self.labels_node_sketches = {}
        for label in self.labels:
            self.labels_node_sketches[label] = snap.TIntIntPrVH()

        if not seed is None:
            random.seed(seed)
        for NI in graph.Nodes():
            node_id = NI.GetId()
            self.node_ids.append(node_id)
            for label in self.labels:
                label_node_sketch = self.labels_node_sketches[label]
                label_node_sketch[node_id] = snap.TIntPrV()
            
            rank = random.uniform(0, 1)
            self.rankings[node_id] = rank            

    def calculate_graph_sketch(self):
        # pool = multiprocessing.Pool()
        # sketch_label = []
        # for label in self.labels:
        #     sketch_label.append(label, self.labels_node_sketches[label])
        # imap_unordered_it = pool.imap_unordered(self.calculate_graph_sketch_label, sketch_label)
        # for _ in tqdm.tqdm(imap_unordered_it, total=len(sketch_label)):
        #     pass
        for label in self.labels:
            self.calculate_graph_sketch_label(self.labels_node_sketches[label], label)
            
    # def calculate_graph_sketch_label(self, label_sketch):
    def calculate_graph_sketch_label(self, node_sketches, label):
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
                sketch = node_sketches[parent_node_id]
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
                    node_sketches[parent_node_id].AddSorted(pair, False)

                    # continue bfs
                    NI = self.tranposed_graph.GetNI(parent_node_id)
                    out_node_ids = NI.GetOutEdges()
                    for out_node_id in out_node_ids:
                        if out_node_id not in visited_nodes:
                            edge_id = self.graph.GetEI(
                                out_node_id, parent_node_id).GetId()
                            edge_label = self.graph.GetStrAttrDatE(
                                edge_id, 'EDGE_LABEL')
                            if edge_label == label:
                                visited_nodes.AddKey(out_node_id)
                                distances[out_node_id] = parent_distance + 1
                                queue.append(out_node_id)
            
        # for node_id, sketch in node_sketches.items():
        #     print(node_id, sketch.Len())

    def cardinality_estimation_labels(self, labels, dist=math.inf):
        size_estimate = 0
        for node_id in self.node_ids:
            size_estimate += self.cardinality_estimation_labels_node_id(node_id, labels, dist=dist)
        return size_estimate
    
    def cardinality_estimation_labels_node_id(self, root_node_id, labels, dist=math.inf):
        queue = snap.TIntV()
        queue.append(root_node_id)
        node_id_exclude_label = {}
        node_id_exclude_label[root_node_id] = None
        neighbors = snap.TFltV()
        ranking = self.rankings[root_node_id]
        neighbors.AddSorted(ranking, True)
        
        size_estimate = 0
        while not queue.Empty():
            node_id = queue.pop(0)
            exclude_label = node_id_exclude_label[node_id]
            break_dist = math.inf
            
            for label in labels:
                if label == exclude_label:
                    continue
                sketch = self.labels_node_sketches[label][node_id]
                for pair in sketch:
                    ads_dist = pair.GetVal1()
                    ads_node_id = pair.GetVal2()
                    if ads_dist > break_dist:
                        break
                    if not ads_node_id in node_id_exclude_label:
                        node_id_exclude_label[ads_node_id] = label
                        queue.append(ads_node_id)
                        ads_ranking = self.rankings[ads_node_id]
                        neighbors.AddSorted(ads_ranking, True)
                    elif ads_node_id != node_id:
                        break_dist = ads_dist
                    
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
