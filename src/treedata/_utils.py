import networkx as nx


def get_leaves(tree: nx.DiGraph) -> list[str]:
    """Get the leaves of a tree."""
    leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]
    return leaves


def get_root(tree: nx.DiGraph) -> str:
    """Get the root of a tree."""
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree has {len(roots)} roots")
    return roots[0]
