from collections import deque

import networkx as nx
import numpy as np
import pandas as pd


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


def digraph_to_dict(G: nx.DiGraph) -> dict:
    """Convert a networkx.DiGraph to a dictionary."""
    G = nx.DiGraph(G)
    edge_dict = nx.to_dict_of_dicts(G)
    # Get node data
    node_dict = {node: G.nodes[node] for node in G.nodes()}
    # Combine edge and node data in one dictionary
    graph_dict = {"edges": edge_dict, "nodes": node_dict}

    return graph_dict


def dict_to_digraph(graph_dict: dict) -> nx.DiGraph:
    """Convert a dictionary to a networkx.DiGraph."""
    G = nx.DiGraph()
    # Add nodes and their attributes
    for node, attrs in graph_dict["nodes"].items():
        G.add_node(node, **attrs)
    # Add edges and their attributes
    for source, targets in graph_dict["edges"].items():
        for target, attrs in targets.items():
            G.add_edge(source, target, **attrs)
    return G


def make_serializable(data) -> dict:
    """Make a graph dictionary serializable."""
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list | tuple | set):
        return [make_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic | np.number):
        return data.item()
    elif isinstance(data, pd.Series):
        return data.tolist()
    else:
        return data
