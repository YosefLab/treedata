import networkx as nx
import pytest

from treedata._utils import get_leaves, get_root, subset_tree


@pytest.fixture
def tree():
    tree = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    root = get_root(tree)
    depths = nx.single_source_shortest_path_length(tree, root)
    nx.set_node_attributes(tree, values=depths, name="depth")
    yield tree


def test_get_leaves():
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "0"), ("root", "1")])
    assert get_leaves(tree) == ["0", "1"]


def test_get_root():
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "0"), ("root", "1")])
    assert get_root(tree) == "root"


def test_get_root_raises():
    # Has cycle
    has_cycle = nx.DiGraph()
    has_cycle.add_edges_from([("root", "0"), ("0", "root")])
    with pytest.raises(ValueError):
        get_root(has_cycle)
    # Multiple roots
    multi_root = nx.DiGraph()
    multi_root.add_edges_from([("root", "0"), ("bad", "0")])
    with pytest.raises(ValueError):
        get_root(multi_root)


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
