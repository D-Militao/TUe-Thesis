from time import time
import pandas as pd
import tracemalloc

import snap

from .context import snap_util, load_data
from .context import GraphSketch
from .test_util import elapsed_time, elapsed_time_str


def test_time_all_distance_sketch_node_ids(network, node_ids, k):
    # Create the sketch for the given k
    start_sketch = time()
    graph_sketch = GraphSketch(network, k)
    graph_sketch.calculate_graph_sketch()
    sketch_construction_time = elapsed_time(start_sketch)

    # Perform estimates on the sketch
    estimates = []  # snap.TFltV()
    start_est = time()
    for node_id in node_ids:
        estimates.append(graph_sketch.cardinality_estimation_node_id(node_id))
    total_estimation_time = elapsed_time(start_est)

    return sketch_construction_time, total_estimation_time, estimates


def test_memory_all_distance_sketch_node_ids(network, node_ids, k):
    # Create the sketch for the given k
    before_sketch, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    graph_sketch = GraphSketch(network, k)
    graph_sketch.calculate_graph_sketch()
    after_sketch, sketch_peak = tracemalloc.get_traced_memory()
    sketch_size = after_sketch-before_sketch

    # Perform estimates on the sketch
    estimates = []  # snap.TFltV()
    before_est, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    for node_id in node_ids:
        estimates.append(graph_sketch.cardinality_estimation_node_id(node_id))
    after_est, est_peak = tracemalloc.get_traced_memory()
    total_estimation_size = after_est-before_est

    return sketch_size, sketch_peak, total_estimation_size, est_peak, estimates
