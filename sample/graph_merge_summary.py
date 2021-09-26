import itertools
import snap

from dataclasses import dataclass
from itertools import count, groupby


from constants import Constants
import util


def init_evaluation_network(edge_labels):
    evaluation_network = snap.TNEANet.New()
    # Node attributes
    evaluation_network.AddFltAttrN(Constants.SN_NODE_WEIGHT.value)
    evaluation_network.AddFltAttrN(Constants.SN_EDGE_WEIGHT.value)
    for label in edge_labels:
        evaluation_network.AddFltAttrN(
            Constants.SN_LABEL_PERCENTAGE.value+label)
        evaluation_network.AddFltAttrN(Constants.SN_LABEL_REACH.value + label)
    evaluation_network.AddIntAttrN(Constants.META_NODE_ID.value)

    # Edge attributes
    evaluation_network.AddStrAttrE(Constants.EDGE_LABEL.value)
    evaluation_network.AddFltAttrE(Constants.SE_EDGE_WEIGHT.value)
    return evaluation_network


def init_merge_network():
    merge_network = snap.TNEANet.New()
    return merge_network


@dataclass
class MetaNode:
    """Class for keeping data on an aggregation of nodes."""
    id: int
    grouping_label: str
    sub_network: snap.TNEANet


class GraphMergeSummary:
    def __init__(self, network):
        self.network = network
        self.network.AddIntAttrN(Constants.META_NODE_ID.value)
        self.edge_labels = util.get_edges_attribute_values(
            network, Constants.EDGE_LABEL.value)

        self.evaluation_network = init_evaluation_network(self.edge_labels)
        self.merge_network = init_merge_network()

    def compute_groupings(self):
        """Returns the groupings on the original network."""
        groupings = {}
        labels_edge_ids = util.get_edge_ids_per_attribute_value(
            self.network, Constants.EDGE_LABEL.value)

        node_ids_in_groupings = snap.TIntV()
        for label, edge_ids in sorted(labels_edge_ids.items(),
                                      key=lambda item: len(item[1]),
                                      reverse=True):
            grouping = self.network.ConvertESubGraph(snap.TNEANet, edge_ids)
            # We cannot use DelNodes; an exception is thrown if we try to
            # remove a node that is not there. There is no way to continue
            # after the exception is thrown, hence not all nodes get removed.
            for node_id in node_ids_in_groupings:
                try:
                    grouping.DelNode(node_id)
                except Exception:  # CHECK specify the exception that is thrown
                    pass
            for NI in grouping.Nodes():
                node_ids_in_groupings.append(NI.GetId())
            if not grouping.Empty():
                groupings[label] = grouping

        # We must place nodes with degree zero in their own grouping
        zero_deg_node_ids = util.get_zero_degree_node_ids(self.network)
        if not zero_deg_node_ids.Empty():
            grouping = self.network.ConvertSubGraph(
                snap.TNEANet, zero_deg_node_ids)
            groupings[Constants.NO_LABEL.value] = grouping

        return groupings

    def create_meta_node(self, network, meta_node_id, grouping_label, inner_node_ids):
        """Returns a MetaNode and registers it in the network it corresponds to.
        
        Keyword arguments:
        network -- a snap.TNEANet network that the MetaNode is aggregating 
        nodes of
        meta_node_id -- the id of the MetaNode
        grouping_label -- the label that the nodes that are part of the 
        MetaNode have in common
        inner_node_ids -- the ids of the nodes in network that MetaNode is 
        aggregating
        """
        sub_network = network.ConvertSubGraph(snap.TNEANet, inner_node_ids)
        meta_node = MetaNode(meta_node_id, grouping_label, sub_network)

        # Add an attribute to the nodes in the original network indicating the
        # the meta node they are a part of
        for node_id in inner_node_ids:
            network.AddIntAttrDatN(node_id, meta_node_id,
                                   Constants.META_NODE_ID.value)

        return meta_node

    # CHECK: Perhaps rename to compute_super_nodes
    def compute_subgroupings(self, groupings):
        """Returns the subgroupings of the original graph."""
        super_node_id_counter = itertools.count(start=1)
        super_nodes = {}
        for grouping_label, grouping in groupings.items():
            wccs = grouping.GetWccs()
            for wcc in wccs:
                # Collect the node ids into a TIntV
                inner_node_ids = snap.TIntV()
                for node_id in wcc:
                    inner_node_ids.Add(node_id)

                # Get the super node id
                super_node_id = next(super_node_id_counter)

                # Create super node
                super_node = self.create_meta_node(
                    self.network, super_node_id, grouping_label, inner_node_ids)

                # Store super node
                super_nodes[super_node_id] = super_node

        return super_nodes

    def calculate_compression_attributes(self, meta_network, meta_node):
        """Calculates compression attributes, node weight and edge weight, 
        which are number of inner nodes and inner edges of the given meta_node.
        The calculate attributes are added to the given meta_network.        
        """
        node_weight = meta_node.sub_network.GetNodes()
        meta_network.AddFltAttrDatN(
            meta_node.id, node_weight, Constants.NODE_WEIGHT.value)

        # Number of edges inside super node
        edge_weight = meta_node.sub_network.GetEdges()
        meta_network.AddFltAttrDatN(
            meta_node.id, edge_weight, Constants.EDGE_WEIGHT.value)

        return node_weight, edge_weight

    def calculate_inner_connectivity_attributes(self, super_node, edge_weight):
        """Calculates inner connectivity attributes for a given super node."""
        labels_to_inner_edge_ids = {label: snap.TIntV()
                                    for label in self.edge_labels}

        # Collect the inner edge id per edge label
        for EI in super_node.sub_network.Edges():
            src_node_id = EI.GetSrcNId()
            dst_node_id = EI.GetDstNId()
            edge_id = self.network.GetEI(src_node_id, dst_node_id).GetId()
            # We need to get the label directly from network because
            # sub_network does not point to the attributes
            label = self.network.GetStrAttrDatE(
                edge_id, Constants.EDGE_LABEL.value)
            labels_to_inner_edge_ids[label].append(edge_id)

        # Store label frequency percentage and calculate reachability
        for label, edge_ids in labels_to_inner_edge_ids.items():
            try:
                percentage = edge_ids.Len() / edge_weight
            except ZeroDivisionError:
                percentage = 0
            self.evaluation_network.AddFltAttrDatN(super_node.id, percentage,
                                                 Constants.LABEL_PERCENTAGE.value + label)

            # For each edge label we want to know how many connection inside
            # the super node exist using only that edge label.
            reach = 0
            # if edge_ids.Len() > 0:
            label_sub_network_sn = self.network.ConvertESubGraph(
                snap.TNEANet, edge_ids)
            for NI in label_sub_network_sn.Nodes():
                node_id = NI.GetId()
                bfs_tree = label_sub_network_sn.GetBfsTree(
                    node_id, True, False)
                reach = reach + bfs_tree.GetEdges()
            self.evaluation_network.AddFltAttrDatN(super_node.id, reach,
                                                 Constants.LABEL_REACH.value + label)

    def calculate_super_node_attributes(self, super_node):
        """Calculates the attributes for the given super node and adds them to 
        the evaluation network
        """
        node_weight, edge_weight = self.calculate_compression_attributes(
            self.evaluation_network, super_node)

        self.calculate_inner_connectivity_attributes(super_node, edge_weight)

        # (3) Outer-Connectivity -- computeConcatenationProperties tthis
        # function does still needs implementation

    def calculate_super_node_edges(self, super_node):
        super_node_connection_counter = {}
        src_super_node_id = super_node.id
        for NI in super_node.sub_network.Nodes():
            src_node_id = NI.GetId()
            original_NI = self.network.GetNI(src_node_id)
            # Go through the outer connections of the node in the original network
            for idx in range(original_NI.GetOutDeg()):
                dst_node_id = original_NI.GetOutNId(idx)
                dst_super_node_id = self.network.GetIntAttrDatN(
                    dst_node_id, Constants.META_NODE_ID.value)
                # if it connects to a node that belongs to a different 
                # super node then count it
                print(src_super_node_id, dst_super_node_id)
                if src_super_node_id != dst_super_node_id:
                    edge_id = self.network.GetEI(
                        src_node_id, dst_node_id).GetId()
                    label = self.network.GetStrAttrDatE(
                        edge_id, Constants.EDGE_LABEL.value)
                    super_node_connection_counter[(dst_super_node_id, label)] = (
                        super_node_connection_counter.setdefault(
                            (dst_super_node_id, label), 0) + 1)
        
        # Add super edges and their attributes
        for (dst_super_node_id, label), edge_weight in super_node_connection_counter.items():
            # Add super edge to evaluation graph
            super_edge_id = self.evaluation_network.AddEdge(
                src_super_node_id, dst_super_node_id)

            # Add label of the super edge
            self.evaluation_network.AddStrAttrDatE(
                super_edge_id, label, Constants.EDGE_LABEL.value)

            # Number of edges inside super edge
            self.evaluation_network.AddFltAttrDatE(
                super_edge_id, edge_weight, Constants.EDGE_WEIGHT.value)

    def build_evalutation_graph(self):
        groupings = self.compute_groupings()
        subgroupings = self.compute_subgroupings(groupings)

        # Add all super nodes to evaluation graph
        for super_node_id in subgroupings.keys():
            self.evaluation_network.AddNode(super_node_id)

        # Compute super node attributes and its super edges
        for super_node_id, super_node in subgroupings.items():
            self.calculate_super_node_attributes(super_node)
            self.calculate_super_node_edges(super_node)
