from collections import deque

import networkx as nx


def subset_tree(tree: nx.DiGraph, leaves: list[str], asview: bool) -> nx.DiGraph:
    """Subset tree."""
    keep_nodes = set(leaves)
    nodes_to_check = deque()
    for node in leaves:
        nodes_to_check.extend(tree.predecessors(node))
    while nodes_to_check:
        node = nodes_to_check.popleft()
        if node in keep_nodes:
            continue
        else:
            keep_nodes.add(node)
            nodes_to_check.extend(tree.predecessors(node))
    if asview:
        return tree.subgraph(keep_nodes)
    else:
        return tree.subgraph(keep_nodes).copy()
