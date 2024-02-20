import networkx as nx
import pytest

from treedata._utils import subset_tree


@pytest.fixture
def tree():
    tree = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    root = [n for n, d in tree.in_degree() if d == 0][0]
    depths = nx.single_source_shortest_path_length(tree, root)
    nx.set_node_attributes(tree, values=depths, name="depth")
    yield tree


def test_subset_tree(tree):
    # copy
    subtree = subset_tree(tree, [7, 8, 9], asview=False)
    expected_edges = [
        (0, 1),
        (1, 3),
        (1, 4),
        (3, 7),
        (3, 8),
        (4, 9),
    ]
    subtree.nodes[9]["depth"] = "new_value"
    assert list(subtree.edges) == expected_edges
    assert tree.nodes[9]["depth"] == 3
    assert subtree.nodes[9]["depth"] == "new_value"
    # view
    subtree = subset_tree(tree, [7, 8, 9], asview=True)
    subtree.nodes[9]["depth"] = "new_value"
    assert list(subtree.edges) == expected_edges
    assert tree.nodes[9]["depth"] == "new_value"
    assert subtree.nodes[9]["depth"] == "new_value"
