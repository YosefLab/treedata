from collections import deque
from typing import Any, Literal

import networkx as nx


def subset_tree(
    tree: nx.DiGraph, nodes: list[str | int] | set[str | int], asview: bool, alignment: str = "leaves"
) -> nx.DiGraph:
    """Subset tree."""
    keep_nodes = set(nodes)
    # if leaves add all ancestors to keep_nodes
    if alignment == "leaves":
        nodes_to_check = deque()
        for node in nodes:
            nodes_to_check.extend(tree.predecessors(node))
        while nodes_to_check:
            node = nodes_to_check.popleft()
            if node in keep_nodes:
                continue
            else:
                keep_nodes.add(node)
                nodes_to_check.extend(tree.predecessors(node))
    subgraph = tree.subgraph(keep_nodes)
    if len(keep_nodes) > 0 and not nx.is_tree(subgraph):
        raise ValueError("Subset is not a tree. Please check your input.")
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


def _get_nodes(trees: dict[str, nx.DiGraph], alignment: str | None) -> list[Any]:
    """Get nodes from trees."""
    nodes = set()
    for _, tree in trees.items():
        if alignment == "leaves":
            nodes.update({node for node in tree.nodes() if tree.out_degree(node) == 0})
        else:
            nodes.update(tree.nodes())
    return list(nodes)
