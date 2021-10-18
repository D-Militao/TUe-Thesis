import os
import sys
from time import time

sys.path.insert(0, os.path.abspath('..'))

from snap_util import snap_util, load_data
from graph_merge_summary import GraphMergeSummary
from all_distance_sketch import GraphSketch
from estimation_function import estimation_function

__start_time__ = time()