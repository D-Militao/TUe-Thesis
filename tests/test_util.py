from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
from time import time, strftime, gmtime

from .context import __start_time__


class TestTracker:
    def __init__(self):
        self.start()

    def start(self):
        """Starts the tracker."""
        self.time = time()

    def track(self):
        """Returns time elapsed, memory increase and memory increase peak 
        since last time start was called.
        """
        elapsed_time = time() - self.time
        return elapsed_time


def stopwatch() -> str:
    return strftime("%H:%M:%S", gmtime(time() - __start_time__))


def test_print(message):
    print(f'[{current_time_str()}] - [{stopwatch()}] {message}')


def elapsed_time_str(start_time: float) -> str:
    """Returns the elapsed time since the given start time as a string."""
    return strftime("%H:%M:%S", gmtime(time() - start_time))


def elapsed_time(start_time: float) -> float:
    """Returns the elapsed time since the given start time."""
    return (time() - start_time)


def current_date_time_str():
    return strftime('%Y-%m-%d_%Hh%Mm%Ss', gmtime(time()))


def current_time_str():
    return strftime('%Hh%Mm%Ss', gmtime(time()))


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    def dict_handler(d): return chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def size_transitive_closure(network):
    size = 0
    for NI in network.Nodes():
        node_id = NI.GetId()
        bfs_tree = network.GetBfsTree(node_id, True, False)
        # NOTE We are counting a nodes connection to itself even if no such edge exists
        size += bfs_tree.GetNodes()
    return size


def size_transitive_closure_node_ids(network, node_ids):
    size = 0
    for node_id in node_ids:
        bfs_tree = network.GetBfsTree(node_id, True, False)
        # NOTE We are counting a nodes connection to itself even if no such edge exists
        size += bfs_tree.GetNodes()
    return size


def size_transitive_closure_node_pairs(network, node_id_pairs):
    size = 0
    for (src_node_id, dst_node_id) in node_id_pairs:
        bfs_tree = network.GetBfsTree(src_node_id, True, False)
        for NI in bfs_tree.Nodes():
            node_id = NI.GetId()
            if node_id == dst_node_id:
                size += 1
                break
    return size


def size_transitive_closure_node_sets(network, src_nodes, dst_nodes):
    size = 0
    for src_node_id in src_nodes:
        bfs_tree = network.GetBfsTree(src_node_id, True, False)
        for NI in bfs_tree.Nodes():
            node_id = NI.GetId()
            if node_id in dst_nodes:
                size += 1
    return size
