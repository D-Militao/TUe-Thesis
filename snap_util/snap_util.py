from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import snap

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    def dict_handler(d): return chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
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


def get_edge_ids_per_attribute_value(network, attribute_name):
    """Returns a dictionary of edge ids per attribute value for a given 
    attribute name.

    Keyword arguments:
    network -- a snap.TNEANet network
    attribute_name -- the name of the attribute we want the values of
    """
    attribute_values_edge_ids = {}
    for EI in network.Edges():
        edge_id = EI.GetId()
        attribute_values = network.GetStrAttrDatE(EI, attribute_name)
        attribute_values_edge_ids.setdefault(
            attribute_values, snap.TIntV()).append(edge_id)
    return attribute_values_edge_ids


def get_edges_attribute_values(network, attribute_name):
    """Returns a snap vector with the attribute values for a given attribute 
    name.

    Keyword arguments:
    network -- a snap.TNEANet network
    attribute_name -- the name of the attribute we want the values of
    """
    attribute_values = snap.TStrV()
    for EI in network.Edges():
        label_value = network.GetStrAttrDatE(EI, attribute_name)
        attribute_values.AddUnique(label_value)
    return attribute_values


def get_zero_degree_node_ids(network):
    """Returns a snap vector with the ids of the zero degree nodes."""
    zero_deg_node_ids = snap.TIntV()
    for NI in network.Nodes():
        if NI.GetDeg() == 0:
            zero_deg_node_ids.appent(NI.GetId())
    return zero_deg_node_ids


def print_type_attributes(network, id, attr_name_func, attr_value_func):
    attr_names = snap.TStrV()
    getattr(network, attr_name_func)(id, attr_names)
    for attr_name in attr_names:
        val = getattr(network, attr_value_func)(id, attr_name)
        print("--> {}: {}".format(attr_name, val))


def print_all_edge_attributes(network):
    for EI in network.Edges():
        edge_id = EI.GetId()
        src_node_id = EI.GetSrcNId()
        dst_node_id = EI.GetDstNId()
        print("Edge id: {}; ({}) -> ({})".format(
            edge_id, src_node_id, dst_node_id))
        print_type_attributes(network, edge_id, "IntAttrNameEI",
                              "GetIntAttrDatE")
        print_type_attributes(network, edge_id, "FltAttrNameEI",
                              "GetFltAttrDatE")
        print_type_attributes(network, edge_id, "StrAttrNameEI",
                              "GetStrAttrDatE")


def print_all_node_attributes(network):
    for NI in network.Nodes():
        node_id = NI.GetId()
        print("Node id: {}".format(node_id))
        print_type_attributes(network, node_id, "IntAttrNameNI",
                              "GetIntAttrDatN")
        print_type_attributes(network, node_id, "FltAttrNameNI",
                              "GetFltAttrDatN")
        print_type_attributes(network, node_id, "StrAttrNameNI",
                              "GetStrAttrDatN")

def transpose_graph(graph):
    transposed_graph = snap.TNGraph.New()
    for NI in graph.Nodes():
        node_id = NI.GetId()
        transposed_graph.AddNode(node_id)
    for EI in graph.Edges():
        src_node_id = EI.GetSrcNId()
        dst_node_id = EI.GetDstNId()
        transposed_graph.AddEdge(dst_node_id, src_node_id)
    return transposed_graph