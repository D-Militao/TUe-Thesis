from enum import Enum, auto

import snap

from constants import Constants


class EdgeFileColumns(Enum):
    SRC_COL = auto()
    DST_COL = auto()
    SRC_NODE = auto()
    DST_NODE = auto()
    EDGE = auto()


class GMarkUseCase(Enum):
    shop = auto()
    social = auto()
    test = auto()
    uniprot = auto()


# Generic funtion to read edge file into a network
# TODO:
# allow any graph type not only snap.TNEANet
# check if binary exists, if so, load that (rename function to make graph/net)
def edge_file_to_network(filename, ordered_attributes, tab_separated=False, has_title_line=snap.TBool(False), dump=False):
    context = snap.TTableContext()

    src_node_attr_v = snap.TStrV()
    dst_node_attr_v = snap.TStrV()
    edge_attr_v = snap.TStrV()
    schema = snap.Schema()
    for attr_category, (attr_name, attr_type) in ordered_attributes.items():
        if attr_category is EdgeFileColumns.SRC_COL:
            src_col = attr_name
        elif attr_category is EdgeFileColumns.DST_COL:
            dst_col = attr_name
        elif attr_category is EdgeFileColumns.SRC_NODE:
            src_node_attr_v.Add(attr_name)
        elif attr_category is EdgeFileColumns.DST_NODE:
            dst_node_attr_v.Add(attr_name)
        elif attr_category is EdgeFileColumns.EDGE:
            edge_attr_v.Add(attr_name)
        schema.Add(snap.TStrTAttrPr(attr_name, attr_type))

    if tab_separated:
        separator = '\t'
    else:
        separator = ' '

    table = snap.TTable.LoadSS(
        schema, filename, context, separator, has_title_line)

    # network will be an object of type snap.TNEANet
    network = table.ToNetwork(snap.TNEANet, src_col, dst_col,
                              src_node_attr_v, dst_node_attr_v,
                              edge_attr_v, snap.aaFirst)

    if dump:
        network.Dump()

    # Save to binary
    outfile = filename + ".bin"
    FOut = snap.TFOut(outfile)
    table.Save(FOut)
    FOut.Flush()

    return network


def make_residence_hall_network(dump=False):
    filename = "data/moreno_oz/out.moreno_oz_oz"

    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt),
        EdgeFileColumns.EDGE: (Constants.EDGE_LABEL.value, snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info, dump)


def make_pg_paper_network(dump=False):
    filename = "data/example/pg_paper.txt"

    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt),
        EdgeFileColumns.EDGE: (Constants.EDGE_LABEL.value, snap.atStr)
    }

    return edge_file_to_network(filename, edge_file_column_info, dump)


def make_author_repo_network(gmark_use_case, dump=False):
    if gmark_use_case is GMarkUseCase.shop:
        filename = "data/author_repo/shop_dataset.txt"
    elif gmark_use_case is GMarkUseCase.social:
        filename = "data/author_repo/social_network_dataset.txt"
    elif gmark_use_case is GMarkUseCase.test:
        filename = "data/author_repo/test_dataset.txt"
    elif gmark_use_case is GMarkUseCase.uniprot:
        filename = "data/author_repo/uniprot_dataset.txt"

    # dict maintains insertion order
    # add the attributes according to the order they are in their file
    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.EDGE: (Constants.EDGE_LABEL.value, snap.atStr),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info, dump)


def make_gmark_network(gmark_use_case, size=1000, dump=False):
    if gmark_use_case is GMarkUseCase.shop:
        filename = f"data/gmark/shop/shop-graph-{size}.txt"
    elif gmark_use_case is GMarkUseCase.social:
        filename = f"data/gmark/social/social-graph-{size}.txt"
    elif gmark_use_case is GMarkUseCase.test:
        filename = f"data/gmark/test/test-graph-{size}.txt"
    elif gmark_use_case is GMarkUseCase.uniprot:
        filename = f"data/gmark/uniprot/uniprot-graph-{size}.txt"

    # dict maintains insertion order
    # add the attributes according to the order they are in their file
    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.EDGE: (Constants.EDGE_LABEL.value, snap.atStr),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info, dump)
