from collections import deque

import networkx as nx


def get_leaves(tree: nx.DiGraph) -> list[str]:
    """Get the leaves of a tree."""
    leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]
    return leaves


def get_root(tree: nx.DiGraph) -> str:
    """Get the root of a tree."""
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}.")
    return roots[0]


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
