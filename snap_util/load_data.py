from enum import Enum, auto
from os import path

import snap

# How we identify which edge attribute constitutes the edge label
__edge_label__ = "EDGE_LABEL"

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
def edge_file_to_network(filename, ordered_attributes, tab_separated=False, has_title_line=snap.TBool(False)):
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

    # If binary file already exists, then load from it (much faster)
    bin_filename = filename + ".bin"
    if path.exists(bin_filename):
        FIn = snap.TFIn(bin_filename)
        table = snap.TTable.Load(FIn, context)
    else:
        with open(filename,'r') as file:
            for line in file:
                # check if the current line starts with "#"
                if line.startswith("#") or line.startswith('%'):
                    continue
                else:
                    if line.count('\t') > 0:
                        separator = '\t'
                        break
                    elif line.count(' ') > 0:
                        separator = ' '
                        break
                    else:
                        raise ValueError(f'The edge file ({filename}) must be tab or whitespace separated.')
            else:
                # We've reached the EOF without detecting any valid lines
                raise ValueError(f'The file ({filename}) must be an edge file.')

        table = snap.TTable.LoadSS(
            schema, filename, context, separator, has_title_line)
        
        # Save table to binary
        FOut = snap.TFOut(bin_filename)
        table.Save(FOut)
        FOut.Flush()

    # network will be an object of type snap.TNEANet
    network = table.ToNetwork(snap.TNEANet, src_col, dst_col,
                              src_node_attr_v, dst_node_attr_v,
                              edge_attr_v, snap.aaFirst)

    return network


def load_residence_hall_network(dump=False):
    filename = "data/moreno_oz/out.moreno_oz_oz"

    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt),
        EdgeFileColumns.EDGE: (__edge_label__, snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info)


def load_pg_paper_network(dump=False):
    filename = "data/example/pg_paper.txt"

    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt),
        EdgeFileColumns.EDGE: (__edge_label__, snap.atStr)
    }

    return edge_file_to_network(filename, edge_file_column_info)


def load_author_repo_network(gmark_use_case, dump=False):
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
        EdgeFileColumns.EDGE: (__edge_label__, snap.atStr),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info)


def load_gmark_network(gmark_use_case, size=1000, dump=False):
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
        EdgeFileColumns.EDGE: (__edge_label__, snap.atStr),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info)


def load_unlabeled_edge_file(filename):
    """
    Load unlabeled edge files where each row has two integers, 
    the source node id and the destination node id.
    """
    edge_file_column_info = {
        EdgeFileColumns.SRC_COL: ('SRC_COL', snap.atInt),
        EdgeFileColumns.DST_COL: ('DST_COL', snap.atInt)
    }

    return edge_file_to_network(filename, edge_file_column_info, tab_separated=True)
