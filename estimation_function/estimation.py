import math
import pandas as pd
import networkx as nx

import snap


def estimation_function(network, s=None, t=None):
    n = network.GetNodes()
    m = network.GetEdges()
    if s is None:
        s = n
    if t is None:
        t = n

    saturation = (-(m - n) / n)
    estimation = s * t * (1 - math.exp(saturation))
    return estimation
