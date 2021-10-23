import itertools
import snap

from dataclasses import dataclass

from .constants import Constants
from snap_util import snap_util

@dataclass
class MetaNode:
    """Class for keeping data on an aggregation of nodes."""
    id: int
    grouping_label: str
    sub_network: snap.TNEANet


class GraphMergeSummary:
    def __init__(self, network, is_labeled=True):
        self.network = network
        self.network.AddIntAttrN(Constants.META_NODE_ID)
        self.is_labeled = is_labeled
        if self.is_labeled:
            self.edge_labels = snap_util.get_edges_attribute_values(
                network, Constants.EDGE_LABEL)
        else:
            self.edge_labels = snap.TStrV()
            self.edge_labels.append(Constants.UNLABELED)

            self.network.AddStrAttrE(Constants.EDGE_LABEL)
            for EI in self.network.Edges():
                edge_id = EI.GetId()
                self.network.AddStrAttrDatE(
                    edge_id, Constants.UNLABELED, Constants.EDGE_LABEL)

        self.init_evaluation_network()
        self.init_merge_network()

    def init_evaluation_network(self):
        self.evaluation_network = snap.TNEANet.New()

        # Node attributes
        self.evaluation_network.AddFltAttrN(Constants.NODE_WEIGHT)
        self.evaluation_network.AddFltAttrN(Constants.EDGE_WEIGHT)
        for label in self.edge_labels:
            self.evaluation_network.AddFltAttrN(
                Constants.LABEL_PERCENTAGE+label)
            self.evaluation_network.AddFltAttrN(Constants.LABEL_REACH + label)
        self.evaluation_network.AddIntAttrN(Constants.META_NODE_ID)

        # Edge attributes
        self.evaluation_network.AddStrAttrE(Constants.EDGE_LABEL)
        self.evaluation_network.AddFltAttrE(Constants.EDGE_WEIGHT)

    def init_merge_network(self):
        self.merge_network = snap.TNEANet.New()

        # Node attributes
        self.merge_network.AddStrAttrE(Constants.AVG_NODE_WEIGHT)

        # Edge Attributes
        self.merge_network.AddStrAttrE(Constants.EDGE_LABEL)

    def compute_node_groups(self):
        """Returns the groups on the original network."""
        groups = {}
        labels_edge_ids = snap_util.get_edge_ids_per_attribute_value(
            self.network, Constants.EDGE_LABEL)

        node_ids_in_groups = snap.TIntV()
        for label, edge_ids in sorted(labels_edge_ids.items(),
                                      key=lambda item: len(item[1]),
                                      reverse=True):
            group = self.network.ConvertESubGraph(snap.TNEANet, edge_ids)
            # We cannot use DelNodes; an exception is thrown if we try to
            # remove a node that is not there. There is no way to continue
            # after the exception is thrown, hence not all nodes get removed.
            for node_id in node_ids_in_groups:
                try:
                    group.DelNode(node_id)
                except Exception:  # TODO specify the exception that is thrown
                    pass
            for NI in group.Nodes():
                node_ids_in_groups.append(NI.GetId())
            if not group.Empty():
                groups[label] = group

        # We must place nodes with degree zero in their own group
        zero_deg_node_ids = snap_util.get_zero_degree_node_ids(self.network)
        if not zero_deg_node_ids.Empty():
            group = self.network.ConvertSubGraph(
                snap.TNEANet, zero_deg_node_ids)
            group[Constants.NO_LABEL] = group

        return groups

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
                                   Constants.META_NODE_ID)

        return meta_node

    def compute_super_nodes(self, groupings):
        super_node_id_counter = itertools.count(start=1)
        super_nodes = {}
        for grouping_label, grouping in groupings.items():
            if self.is_labeled:
                components = grouping.GetWccs()
            else:
                components = grouping.GetSccs()
            for component in components:
                # Collect the node ids into a TIntV
                inner_node_ids = snap.TIntV()
                for node_id in component:
                    inner_node_ids.Add(node_id)

                # Get the super node id
                super_node_id = next(super_node_id_counter)

                # Create super node
                super_node = self.create_meta_node(
                    self.network, super_node_id, grouping_label, inner_node_ids)

                # Store super node
                super_nodes[super_node_id] = super_node

        return super_nodes

    def compute_evaluation_compression_attributes(self, super_node):
        """Computes compression attributes, node weight and edge weight, 
        which are number of inner nodes and inner edges of the given super node.
        """
        node_weight = super_node.sub_network.GetNodes()
        self.evaluation_network.AddFltAttrDatN(
            super_node.id, node_weight, Constants.NODE_WEIGHT)

        # Number of edges inside super node
        edge_weight = super_node.sub_network.GetEdges()
        self.evaluation_network.AddFltAttrDatN(
            super_node.id, edge_weight, Constants.EDGE_WEIGHT)

        return node_weight, edge_weight

    def compute_evaluation_inner_connectivity_attributes(self, super_node, edge_weight):
        """Computes inner connectivity attributes for a given super node."""
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
                edge_id, Constants.EDGE_LABEL)
            labels_to_inner_edge_ids[label].append(edge_id)

        # Store label frequency percentage and calculate reachability
        for label, edge_ids in labels_to_inner_edge_ids.items():
            try:
                percentage = edge_ids.Len() / edge_weight
            except ZeroDivisionError:
                percentage = 0
            self.evaluation_network.AddFltAttrDatN(
                super_node.id,
                percentage,
                Constants.LABEL_PERCENTAGE + label)

            # For each edge label we want to know how many connection inside
            # the super node exist using only that edge label.
            reach = 0
            # if edge_ids.Len() > 0:
            label_sub_network_sn = self.network.ConvertESubGraph(
                snap.TNEANet, edge_ids)
            for NI in label_sub_network_sn.Nodes():
                node_id = NI.GetId()
                # TODO doing a bfs search for every node in order to calculate the reach is incredibly inefficient
                bfs_tree = label_sub_network_sn.GetBfsTree(
                    node_id, True, False)
                reach = reach + bfs_tree.GetEdges()
            self.evaluation_network.AddFltAttrDatN(
                super_node.id, reach, Constants.LABEL_REACH + label)
        
        total_reach = 0
        for NI in super_node.sub_network.Nodes():
            node_id = NI.GetId()
            # TODO doing a bfs search for every node in order to calculate the reach is incredibly inefficient
            bfs_tree = super_node.sub_network.GetBfsTree(node_id, True, False)
            total_reach = total_reach + bfs_tree.GetEdges()
        self.evaluation_network.AddFltAttrDatN(
                super_node.id, total_reach, Constants.TOTAL_REACH)

    def compute_super_node_attributes(self, super_node):
        """Computes the attributes for the given super node and adds them to 
        the evaluation network.
        """
        _, edge_weight = self.compute_evaluation_compression_attributes(
            super_node)

        self.compute_evaluation_inner_connectivity_attributes(
            super_node, edge_weight)

        # (3) Outer-Connectivity -- computeConcatenationProperties this
        # function does still needs implementation

    def compute_super_node_edges(self, super_node):
        """Computes super edges and their attributes for a given super node."""
        super_node_connection_counter = {}
        src_super_node_id = super_node.id
        for NI in super_node.sub_network.Nodes():
            src_node_id = NI.GetId()
            original_NI = self.network.GetNI(src_node_id)
            # Go through the outer connections of the node in the original network
            for idx in range(original_NI.GetOutDeg()):
                dst_node_id = original_NI.GetOutNId(idx)
                dst_super_node_id = self.network.GetIntAttrDatN(
                    dst_node_id, Constants.META_NODE_ID)
                # if it connects to a node that belongs to a different
                # super node then count it
                if src_super_node_id != dst_super_node_id:
                    edge_id = self.network.GetEI(
                        src_node_id, dst_node_id).GetId()
                    label = self.network.GetStrAttrDatE(
                        edge_id, Constants.EDGE_LABEL)
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
                super_edge_id, label, Constants.EDGE_LABEL)

            # Number of edges inside super edge
            self.evaluation_network.AddFltAttrDatE(
                super_edge_id, edge_weight, Constants.EDGE_WEIGHT)

    def build_evalutation_network(self):
        groupings = self.compute_node_groups()

        super_nodes = self.compute_super_nodes(groupings)

        # Add all super nodes to evaluation graph
        for super_node_id in super_nodes.keys():
            self.evaluation_network.AddNode(super_node_id)

        # Compute super node attributes and its super edges
        for super_node_id, super_node in super_nodes.items():
            self.compute_super_node_attributes(super_node)
            self.compute_super_node_edges(super_node)

    def get_highest_reachability(self, super_node_id):
        """Returns the label and value of the highest reachability statistic of
        the super node with the given super_node_id.
        """
        reach_label = Constants.NO_LABEL
        reach_label_value = 0
        attr_names = snap.TStrV()
        self.evaluation_network.FltAttrNameNI(super_node_id, attr_names)

        for attr_name in attr_names:
            if not attr_name.startswith(Constants.LABEL_REACH):
                continue
            attr_value = self.evaluation_network.GetFltAttrDatN(
                super_node_id, attr_name)
            if attr_value > reach_label_value:
                reach_label = attr_name.split(Constants.SEPARATOR)[-1]
                reach_label_value = attr_value

        return reach_label, reach_label_value

    def compute_super_node_groups(self, is_target_merge):
        super_node_ids_groups = {}

        for NI in self.evaluation_network.Nodes():
            super_node_id = NI.GetId()
            degree = NI.GetInDeg() if is_target_merge else NI.GetOutDeg()
            edge_labels = set()  # TODO find better data structure

            # Collect the labels os the in/out edges
            for idx in range(degree):
                super_edge_id = NI.GetInEId(
                    idx) if is_target_merge else NI.GetOutEId(idx)
                label = self.evaluation_network.GetStrAttrDatE(
                    super_edge_id, Constants.EDGE_LABEL)
                edge_labels.add(label)

            # TODO reachability value is not used
            # author tried to also group with it, but no info on results
            reach_label, reach_label_value = self.get_highest_reachability(
                super_node_id)

            super_node_ids_groups.setdefault(
                (reach_label, frozenset(edge_labels)), snap.TIntV()).append(super_node_id)

        return super_node_ids_groups

    def compute_hyper_nodes(self, super_node_groups):
        hyper_node_id_counter = itertools.count(start=1)
        hyper_nodes = {}

        for (reach_label, _), super_node_ids in super_node_groups.items():
            hyper_node_id = next(hyper_node_id_counter)

            hyper_node = self.create_meta_node(
                self.evaluation_network, hyper_node_id, reach_label, super_node_ids)

            hyper_nodes[hyper_node_id] = hyper_node

        return hyper_nodes

    def compute_hyper_node_edges_and_attributes(self, hyper_node, is_target_merge):
        """Computes the attributes for the given hyper node and adds them to 
        the merge network.
        """
        node_attributes = {}
        hyper_edges = {}

        # * Subgraph does not have the node attributes
        for NI in hyper_node.sub_network.Nodes():
            # Sum the node weight among all super nodes
            node_weight = self.evaluation_network.GetFltAttrDatN(
                NI, Constants.NODE_WEIGHT)
            node_attributes[Constants.NODE_WEIGHT] = (
                node_attributes.setdefault(Constants.NODE_WEIGHT, 0) + node_weight)

            # Sum the edge weight among all super nodes
            edge_weight = self.evaluation_network.GetFltAttrDatN(
                NI, Constants.EDGE_WEIGHT)
            node_attributes[Constants.EDGE_WEIGHT] = (
                node_attributes.setdefault(Constants.EDGE_WEIGHT, 0) + edge_weight)

            # Sum the total reach among all super nodes
            total_reach = self.evaluation_network.GetFltAttrDatN(
                NI, Constants.TOTAL_REACH)
            node_attributes[Constants.TOTAL_REACH] = (
                node_attributes.setdefault(Constants.TOTAL_REACH, 0) + total_reach)

            # Now we handle the attributes that relate to each label
            for label in self.edge_labels:
                # Sum the label reach for each label in the super nodes
                label_reach = self.evaluation_network.GetFltAttrDatN(
                    NI, Constants.LABEL_REACH + label)
                node_attributes[Constants.LABEL_REACH + label] = (
                    node_attributes.setdefault(Constants.LABEL_REACH + label, 0) + label_reach)

                # For the percentage of inner edge labels we need to multiply by the total edges
                # TODO: why store the percentage and not just the total?
                label_percent = self.evaluation_network.GetFltAttrDatN(
                    NI, Constants.LABEL_PERCENTAGE + label)
                node_attributes[Constants.LABEL_PERCENTAGE + label] = (
                    node_attributes.setdefault(
                        Constants.LABEL_PERCENTAGE + label, 0) 
                        + (edge_weight * label_percent))

            # TODO Hyper edge stuff check strategy
            original_NI = self.evaluation_network.GetNI(NI.GetId())
            degree = original_NI.GetInDeg() if is_target_merge else original_NI.GetOutDeg()
            # Collect the labels and edge weight of the in/out edges
            for idx in range(degree):
                # Get super edge attributes
                super_edge_id = original_NI.GetInEId(
                    idx) if is_target_merge else original_NI.GetOutEId(idx)
                edge_weight = self.evaluation_network.GetFltAttrDatE(
                    super_edge_id, Constants.EDGE_WEIGHT)
                edge_label = self.evaluation_network.GetStrAttrDatE(
                    super_edge_id, Constants.EDGE_LABEL)

                # Get dst hyper node id
                dst_super_node_id = original_NI.GetInNId(
                    idx) if is_target_merge else original_NI.GetOutNId(idx)
                dst_hyper_node_id = self.evaluation_network.GetIntAttrDatN(
                    dst_super_node_id, Constants.META_NODE_ID)

                # Store hyper edge information
                hyper_edges[(dst_hyper_node_id, edge_label)] = (
                    hyper_edges.setdefault(
                        (dst_hyper_node_id, edge_label), 0) + edge_weight)

        # Now that we have sum of the (edge label percent * edge_weight) of
        # every super node, we divide it by the total_edge_weight (i.e. the
        # edge weight of the hyper node)
        total_edge_weight = node_attributes[Constants.EDGE_WEIGHT]
        for label in self.edge_labels:
            # TODO: why store the percentage and not just the total?
            try:
                percentage = node_attributes[Constants.LABEL_PERCENTAGE + label] / total_edge_weight
            except ZeroDivisionError:
                percentage = 0
            node_attributes[Constants.LABEL_PERCENTAGE + label] = percentage

        # Number of super nodes inside hyper node
        hn_node_weight = hyper_node.sub_network.GetNodes()
        node_attributes[Constants.SUPER_NODE_WEIGHT] = hn_node_weight

        # Average number of nodes inside each super node of the hyper node
        total_node_weight = node_attributes[Constants.NODE_WEIGHT]
        avg_node_weight = total_node_weight / hn_node_weight
        node_attributes[Constants.AVG_NODE_WEIGHT] = avg_node_weight

        # TODO Still need to calculate frontier attributes

        # Add all the properties to the hyper node
        for attr_name, attr_value in node_attributes.items():
            self.merge_network.AddFltAttrDatN(
                hyper_node.id, attr_value, attr_name)

        # Calculate hyper edges and attributes
        for (dst_hyper_node_id, edge_label), sum_edge_weight in hyper_edges.items():
            # Add hyper edge to the merge graph
            hyper_edge_id = self.merge_network.AddEdge(
                hyper_node.id, dst_hyper_node_id)

            # Add label of the hyper edge
            self.merge_network.AddStrAttrDatE(
                hyper_edge_id, label, Constants.EDGE_LABEL)

            # Number of edges inside hyper edge
            self.merge_network.AddFltAttrDatE(
                hyper_edge_id, sum_edge_weight, Constants.EDGE_WEIGHT)

    def build_merge_network_labeled(self, is_target_merge=True):
        # We clear the merge network in case it has already been calculated
        # This allows us to recalculate the merge network with a different merge strategy
        self.merge_network.Clr()
        # Check if there is a way to reverse the edges in the network
        # if so then we can just reverse the graph depending on the strategy
        super_node_groups = self.compute_super_node_groups(is_target_merge)

        hyper_nodes = self.compute_hyper_nodes(super_node_groups)

        # Add all hyper nodes to evaluation graph
        for hyper_node_id in hyper_nodes.keys():
            self.merge_network.AddNode(hyper_node_id)

        # Compute hyper node attributes and its hyper edges
        for hyper_node_id, hyper_node in hyper_nodes.items():
            self.compute_hyper_node_edges_and_attributes(
                hyper_node, is_target_merge)

    def build_merge_network_unlabeled(self, is_target_merge=False):
        self.merge_network.Clr()
        super_node_groups = {}
        for NI in self.evaluation_network.Nodes():
            super_node_id = NI.GetId()
            node_weight = self.evaluation_network.GetFltAttrDatN(
                super_node_id, Constants.NODE_WEIGHT)
            
            out_deg = NI.GetOutDeg()
            out_node_ids = snap.TIntV()
            for out_idx in range(out_deg):
                out_node_id = NI.GetOutNId(out_idx)
                out_node_ids.AddMerged(out_node_id)

            in_deg = NI.GetInDeg()
            in_node_ids = snap.TIntV()
            for in_idx in range(in_deg):
                in_node_id = NI.GetInNId(in_idx)
                in_node_ids.AddMerged(in_node_id)
            
            in_list = []
            for in_node_id in in_node_ids:
                in_list.append(in_node_id)

            out_list = []
            for out_node_id in out_node_ids:
                out_list.append(out_node_id)

            super_node_groups.setdefault(
                (frozenset(in_list), frozenset(out_list)), snap.TIntV()).append(
                    super_node_id)
            
        
        hyper_node_id_counter = itertools.count(start=1)
        hyper_nodes = {}
        for super_node_ids in super_node_groups.values():
            hyper_node_id = next(hyper_node_id_counter)

            hyper_node = self.create_meta_node(
                self.evaluation_network, hyper_node_id, Constants.UNLABELED, super_node_ids)

            hyper_nodes[hyper_node_id] = hyper_node
        
        # Add all hyper nodes to evaluation graph
        for hyper_node_id in hyper_nodes.keys():
            self.merge_network.AddNode(hyper_node_id)

        # Compute hyper node attributes and its hyper edges
        for hyper_node_id, hyper_node in hyper_nodes.items():
            self.compute_hyper_node_edges_and_attributes(
                hyper_node, is_target_merge)

    def build_merge_network(self, is_target_merge=True):
        if self.is_labeled:
            self.build_merge_network_labeled(is_target_merge=is_target_merge)
        else:
            self.build_merge_network_unlabeled(is_target_merge=False)

    def cardinality_estimation_node_id(self, node_id):
        super_node_id = self.network.GetIntAttrDatN(node_id, Constants.META_NODE_ID)
        hyper_node_id = self.evaluation_network.GetIntAttrDatN(super_node_id, Constants.META_NODE_ID)
        hyper_bfs_tree = self.merge_network.GetBfsTree(hyper_node_id, True, False)
        
        size_estimate = self.evaluation_network.GetFltAttrDatN(super_node_id, Constants.NODE_WEIGHT)
        for NI in hyper_bfs_tree.Nodes():
            bfs_node_id = NI.GetId()
            if bfs_node_id == hyper_node_id:
                continue
            node_weight = self.merge_network.GetFltAttrDatN(
                    bfs_node_id, Constants.NODE_WEIGHT)
            super_node_weight = self.merge_network.GetFltAttrDatN(
                    bfs_node_id, Constants.SUPER_NODE_WEIGHT)
            avg_node_weight = self.merge_network.GetFltAttrDatN(
                    bfs_node_id, Constants.AVG_NODE_WEIGHT)
            size_estimate += node_weight
            # size_estimate += self.merge_network.GetFltAttrDatN(
            #     hyper_node_id, Constants.NODE_WEIGHT)
            # for label in self.edge_labels:
            #     size_estimate += self.merge_network.GetFltAttrDatN(
            #         hyper_node_id, Constants.LABEL_REACH+label)
            #     break
            
        return size_estimate

    # def cardinality_estimation_unlabeled_node_id(self, node_id):
    #     super_node_id = self.network.GetIntAttrDatN(node_id, Constants.META_NODE_ID)
    #     super_bfs_tree = self.evaluation_network.GetBfsTree(super_node_id, True, False)
    #     size_estimate = 0
    #     for NI in super_bfs_tree.Nodes():
    #         bfs_node_id = NI.GetId()
    #         node_weight = self.evaluation_network.GetFltAttrDatN(
    #                 bfs_node_id, Constants.NODE_WEIGHT)
    #         super_node_weight = self.evaluation_network.GetFltAttrDatN(
    #                 bfs_node_id, Constants.SUPER_NODE_WEIGHT)
    #         size_estimate += node_weight / super_node_weight
    #         # size_estimate += self.evaluation_network.GetFltAttrDatN(
    #         #         bfs_node_id, Constants.LABEL_REACH+Constants.UNLABELED)
    #     return size_estimate
                
    def check_merge_graph(self):
        se_edge_weight = 0
        for EI in self.evaluation_graph.Edges():
            se_edge_weight += self.valuation_graph.GetFltAttrDatE(
                EI, Constants.SE_EDGE_WEIGHT)

        sn_edge_weight = 0
        for NI in self.devaluation_graph.Nodes():
            sn_edge_weight += self.evaluation_graph.GetFltAttrDatN(
                NI, Constants.SN_EDGE_WEIGHT)

        he_edge_weight = 0
        for EI in self.merge_graph.Edges():
            he_edge_weight += self.merge_graph.GetFltAttrDatE(
                EI, Constants.HE_EDGE_WEIGHT)

        # hn_edge_weight = 0
        # for NI in merge_graph.Nodes():
        #     hn_edge_weight += merge_graph.GetFltAttrDatN(
        #         NI, Constants.HN_EDGE_WEIGHT)

        print(f'{self.network.GetEdges()} -> {se_edge_weight} + {sn_edge_weight} -> {he_edge_weight} + ???')
