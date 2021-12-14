import itertools
import snap

from .constants import Constants
from snap_util import snap_util

class UnlabeledGraphSummary:
    def __init__(self, network):
        self.network = network
        self.evaluation_network = snap.TNEANet.New()
        self.merge_network = snap.TNEANet.New()
        self.s_nodes_inner_nodes = {}
        self.nodes_s_nodes = {}
        self.h_nodes_inner_nodes = {}
        self.nodes_h_nodes = {}
    
    def build_summary(self):
        s_node_id_counter = itertools.count(start=1)
        s_nodes_dst_nodes = {}
        s_nodes_src_nodes = {}
        for scc in self.network.GetSccs():
            # Get the s-node id
            s_node_id = next(s_node_id_counter)
            s_nodes_dst_nodes[s_node_id] = []
            s_nodes_src_nodes[s_node_id] = []
            
            for scc_node_id in scc:
                # Register s-node inner nodes
                self.nodes_s_nodes[scc_node_id] = s_node_id
                self.s_nodes_inner_nodes.setdefault(
                    s_node_id, []).append(scc_node_id)
                
                NI = self.network.GetNI(scc_node_id)
                # Register what nodes the s-node connects to
                dst_node_ids = NI.GetOutEdges()
                for dst_node_id in dst_node_ids:
                    s_nodes_dst_nodes[s_node_id].append(dst_node_id)
                
                # Register what nodes connect to the s-node
                src_node_ids = NI.GetInEdges()
                for src_node_id in src_node_ids:
                    s_nodes_src_nodes[s_node_id].append(src_node_id)
            
            # Create s-node
            self.evaluation_network.AddNode(s_node_id)
        
        # for s_node_id, dst_node_ids in s_nodes_dst_nodes.items():
        #     dst_s_node_ids = []
        #     for dst_node_id in dst_node_ids:
        #         dst_s_node_ids.append(self.nodes_s_nodes[dst_node_id])
        #     dst_s_node_ids.discard(s_node_id)
        #     self.evaluation_network.AddEdge(s_node_id)
        
        h_node_groups = {}
        for s_node_id in self.s_nodes_inner_nodes.keys():
            # Register -nodes that connect to the s_node
            src_s_node_ids = set()
            for src_node_id in s_nodes_src_nodes[s_node_id]:
                src_s_node_ids.add(self.nodes_s_nodes[src_node_id])
            src_s_node_ids.discard(s_node_id)
            
            # Register what s-nodes the s_node connects to
            dst_s_node_ids = set()
            for dst_node_id in s_nodes_dst_nodes[s_node_id]:
                dst_s_node_ids.add(self.nodes_s_nodes[dst_node_id])
            dst_s_node_ids.discard(s_node_id)
            # Group s_nodes into h_nodes based on equal in and out connections
            h_node_groups.setdefault(
                (frozenset(src_s_node_ids), frozenset(dst_s_node_ids)), []).append(s_node_id)
        
        # Create h-nodes
        h_node_id_counter = itertools.count(start=1)
        s_nodes_h_nodes = {}
        for s_node_ids in h_node_groups.values():
            h_node_id = next(h_node_id_counter)
            self.merge_network.AddNode(h_node_id)
            for s_node_id in s_node_ids:
                s_nodes_h_nodes[s_node_id] = h_node_id
                
                for inner_node in self.s_nodes_inner_nodes[s_node_id]:
                    self.h_nodes_inner_nodes.setdefault(
                        h_node_id, []).append(inner_node)
        
        # Create h-edges
        for (src_s_node_ids, dst_s_node_ids), s_node_ids in h_node_groups.items():
            h_node_id = s_nodes_h_nodes[s_node_ids[0]]
            dst_h_nodes = set()
            for dst_s_node_id in dst_s_node_ids:
                dst_h_nodes.add(s_nodes_h_nodes[dst_s_node_id])
            
            for dst_h_node in dst_h_nodes:
                self.merge_network.AddEdge(h_node_id, dst_h_node)

        
    def cardinality_estimation_node_id(self, node_id):
        main_h_node_id = -1
        for h_node_id, inner_nodes in self.h_nodes_inner_nodes.items():
            if node_id in inner_nodes:
                main_h_node_id = h_node_id
        bfs_tree = self.merge_network.GetBfsTree(main_h_node_id, True, False)
        size = 0
        for inner_nodes in self.s_nodes_inner_nodes.values():
            if node_id in inner_nodes:
                size += len(inner_nodes)
                break
            
        for NI in bfs_tree.Nodes():
            bfs_h_node_id = NI.GetId()
            if bfs_h_node_id != main_h_node_id:
                size += len(self.h_nodes_inner_nodes[bfs_h_node_id])
        return size
        
