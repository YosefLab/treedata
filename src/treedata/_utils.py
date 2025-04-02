from collections import deque
from typing import Literal

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


def combine_trees(subsets: list[nx.DiGraph]) -> nx.DiGraph:
    """Combine two or more subsets of a tree into a single tree."""
    # Initialize a new directed graph for the combined tree
    combined_tree = nx.DiGraph()

    # Iterate through each subset and add its nodes and edges to the combined tree
    for subset in subsets:
        combined_tree.add_nodes_from(subset.nodes(data=True))
        combined_tree.add_edges_from(subset.edges(data=True))

    # The combined_tree now contains all nodes and edges from the subsets
    return combined_tree


def _resolve_axis(
    axis: Literal["obs", 0, "var", 1],
) -> tuple[Literal[0], Literal["obs"]] | tuple[Literal[1], Literal["var"]]:
    """Resolve axis argument."""
    if axis in {0, "obs"}:
        return (0, "obs")
    if axis in {1, "var"}:
        return (1, "var")
    raise ValueError(f"`axis` must be either 0, 1, 'obs', or 'var', was {axis}")
