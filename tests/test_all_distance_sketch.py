from time import time
import pandas as pd

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
    # memory occupied

    # Perform estimates on the sketch
    estimates = [] # snap.TFltV()
    start_est = time()
    for node_id in node_ids:
        estimates.append(graph_sketch.cardinality_estimation_node_id(node_id))
    total_estimation_time = elapsed_time(start_est)

    return sketch_construction_time, total_estimation_time, estimates
    