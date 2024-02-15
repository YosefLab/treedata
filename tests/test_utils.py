import networkx as nx
import pytest

from treedata._utils import get_leaves, get_root


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
