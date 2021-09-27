from enum import Enum

# Constants used in the paper:
# NODE_WEIGHT_EVALUATION("NUMBER_OF_NODES_INSIDE_OF_SN"),
# NODE_WEIGHT_FREQUENCY("NUMBER_OF_SN_INSIDE_OF_HN"),
# NODE_WEIGHT_EVALUATION_AVG("AVG_WEIGHT_ON_SN"),
# PERCENTAGE("PERCENTAGE"),
# NUMBER_OF_INNER_EDGES("NUMBER_OF_INNER_EDGES"),
# LABEL("LABEL"),
# EDGE_RATIO("EDGE_RATIO"),
# GROUPING("GROUPING"),
# REACHABILITY_COUNT("REACH_NUMBER_OF_INNER_PATHS"),
# PATH_OUT("REACH_PATH_OUT_BY_LABEL"),                      # Not used in og repo
# PATH_IN("REACH_PATH_IN_BY_LABEL"),                        # Not used in og repo
# EDGE_WEIGHT("EDGE_WEIGHT"),
# PARTICIPATION_LABEL("PARTICIPATION_LABEL"),
# TRAVERSAL_FRONTIERS("TRAVERSAL_FRONTIERS");

class Constants(str, Enum):
    NO_LABEL = "NO_LABEL"
    SEPARATOR = "_"

    # General attributes for all graphs 
    EDGE_LABEL = "EDGE_LABEL" # indicates the edge label, e.g. KNOWS, LIKES...

    # Attribute to add to a network indicating the ID of the node in the aggregate network it corresponds to
    META_NODE_ID = "META_NODE_ID"

    NODE_WEIGHT = "NUMBER_OF_INNER_NODES"
    EDGE_WEIGHT = "NUMBER_OF_INNER_EDGES"

    LABEL_PERCENTAGE = "LABEL_PERCENTAGE_"
    LABEL_REACH = "LABEL_REACH_"

    AVG_NODE_WEIGHT = "AVG_NUMBER_OF_INNER_NODES"
    SUPER_NODE_WEIGHT = "NUMBER_OF_INNER_SUPER_NODES"

    #####################################################
    # BELOW ARE THE CONSTANTS USED PRIOR TO REFACTORING #
    #####################################################

    # # Attributes for the OG graph
    # SUPER_NODE_ID = "SUPER_NODE_ID" # indicates the super node id that the node belongs to
    
    # # Attributes for the Evaluation graph
    # HYPER_NODE_ID = "HYPER_NODE_ID" # indicates the hyper node id that the super node belongs to
    # SN_NODE_WEIGHT = "NUMBER_OF_INNER_NODES" # indicates the number of inner nodes the super node has NOTE: Maybe change this to no_of_inner_base_nodes
    # SN_EDGE_WEIGHT = "NUMBER_OF_INNER_EDGES" # indicates the number of inner edges the super node has NOTE: Maybe change this to no_of_inner_base_edges
    
    # # Inner-Connectivity - Each of the following attributes will exist for each
    # # l-label, i.e., we would have PERCENTAGE_KNOWS, PERCENTAGE_LIKES, for a 
    # # graph with l-labels = {KNOWS, LIKES}.
    # SN_LABEL_PERCENTAGE = "LABEL_PERCENTAGE_" # percent of l-labeled inner edges
    # SN_LABEL_REACH = "LABEL_REACH_" # number of inner node pairs connected with an l-labeled inner edge

    # SE_EDGE_WEIGHT = "SE_NUMBER_OF_EDGES" # indicates the number of inner edges the super edge has
    # # SE_EDGE_RATIO CHECK used by the author to calculate participation, but I don't think it's needed

    # # Attributes for the Merge graph
    # HN_NODE_WEIGHT = "NUMBER_OF_INNER_SUPER_NODES" # indicates the number of inner super nodes the hyper node has
    # HN_AVG_NODE_WEIGHT = "AVG_NUMBER_OF_INNER_NODES" # for a given hyper node, this indicates the average number of inner nodes its super nodes have

    # HE_EDGE_WEIGHT = "HE_NUMBER_OF_EDGES"