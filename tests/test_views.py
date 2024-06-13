import networkx as nx
import numpy as np
import pandas as pd
import pytest

import treedata as td


@pytest.fixture
def tree():
    tree = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    tree = nx.relabel_nodes(tree, {i: str(i) for i in tree.nodes})
    depths = nx.single_source_shortest_path_length(tree, "0")
    nx.set_node_attributes(tree, values=depths, name="depth")
    yield tree


@pytest.fixture
def tdata(tree):
    df = pd.DataFrame({"anno": range(8)}, index=[str(i) for i in range(7, 15)])
    yield td.TreeData(X=np.zeros((8, 8)), obst={"tree": tree}, vart={"tree": tree}, obs=df, var=df, allow_overlap=True)


def test_views(tdata):
    # check that subset is view
    assert tdata[:, 0].is_view
    assert tdata[:2, 0].X.shape == (2, 1)
    tdata_subset = tdata[:2, [0, 1]]
    assert tdata_subset.is_view
    # now transition to actual object
    with pytest.warns(UserWarning):
        tdata_subset.obs["test"] = range(2)
    assert not tdata_subset.is_view
    assert tdata_subset.obs["test"].tolist() == list(range(2))


def test_views_subset_tree(tdata):
    expected_edges = [
        ("0", "1"),
        ("0", "2"),
        ("1", "3"),
        ("2", "5"),
        ("3", "7"),
        ("3", "8"),
        ("5", "11"),
    ]
    # subset with index
    tdata_subset = tdata[[0, 1, 4], :]
    edges = list(tdata_subset.obst["tree"].edges)
    assert edges == expected_edges
    # subset with names
    tdata_subset = tdata[["7", "8", "11"], :]
    edges = list(tdata_subset.obst["tree"].edges)
    assert edges == expected_edges
    # now transition to actual object
    tdata_subset = tdata_subset.copy()
    edges = list(tdata_subset.obst["tree"].edges)
    assert edges == expected_edges
    assert len(tdata.obst["tree"].edges) == 14


def test_views_subset_trees():
    # subset tdata with multiple trees
    tree1 = nx.DiGraph([("root", "0"), ("root", "1")])
    tree2 = nx.DiGraph([("root", "2"), ("root", "3")])
    tdata = td.TreeData(X=np.zeros((8, 8)), allow_overlap=False, obst={"tree1": tree1, "tree2": tree2})
    tdata_subset = tdata[["0", "1"], :]
    assert list(tdata_subset.obst["tree1"].edges) == [("root", "0"), ("root", "1")]
    assert list(tdata_subset.obst["tree2"].edges) == []
    tdata_subset = tdata[["2", "3"], :]
    assert list(tdata_subset.obst["tree1"].edges) == []
    assert list(tdata_subset.obst["tree2"].edges) == [("root", "2"), ("root", "3")]
    tdata_subset = tdata[["0", "1", "2"], :]
    assert list(tdata_subset.obst["tree1"].edges) == [("root", "0"), ("root", "1")]
    assert list(tdata_subset.obst["tree2"].edges) == [("root", "2")]


def test_views_copy():
    # subset tdata with multiple trees
    tree1 = nx.DiGraph([("root", "0"), ("root", "1")])
    tree2 = nx.DiGraph([("root", "2"), ("root", "3")])
    tdata = td.TreeData(X=np.zeros((8, 8)), allow_overlap=False, obst={"tree1": tree1, "tree2": tree2})
    tdata_subset = tdata.copy()[["0", "1"], :].copy()
    assert list(tdata_subset.obst["tree1"].edges) == [("root", "0"), ("root", "1")]
    assert list(tdata_subset.obst["tree2"].edges) == []
    print(tdata_subset)
    tdata_subset = tdata.copy()[["2", "3"], :].copy()
    assert list(tdata_subset.obst["tree1"].edges) == []
    assert list(tdata_subset.obst["tree2"].edges) == [("root", "2"), ("root", "3")]
    tdata_subset = tdata.copy()[["0", "1", "2"], :].copy()
    assert list(tdata_subset.obst["tree1"].edges) == [("root", "0"), ("root", "1")]
    assert list(tdata_subset.obst["tree2"].edges) == [("root", "2")]


def test_views_mutability(tdata):
    # can mutate attributes of graph
    nx.set_node_attributes(tdata.obst["tree"], False, "in_subset")
    subset_leaves = ["7", "8"]
    tdata_subset = tdata[subset_leaves, :]
    nx.set_node_attributes(tdata_subset.obst["tree"], True, "in_subset")
    expected_subset_nodes = ["8", "0", "3", "7", "1"]
    subset_nodes = [
        node for node in tdata_subset.obst["tree"].nodes if tdata_subset.obst["tree"].nodes[node]["in_subset"]
    ]
    assert set(subset_nodes) == set(expected_subset_nodes)
    # cannot mutate structure of graph
    with pytest.raises(nx.NetworkXError):
        tdata_subset.obst["tree"].remove_node("8")


def test_views_set(tdata):
    tdata_subset = tdata[[0, 1, 4], :]
    # bad assignment
    bad_tree = nx.DiGraph()
    bad_tree.add_edge("0", "bad")
    with pytest.raises(ValueError):
        tdata_subset.obst["new_tree"] = bad_tree
    assert tdata_subset.is_view
    # good assignment actualizes object
    new_tree = nx.DiGraph()
    new_tree.add_edge("0", "8")
    with pytest.warns(UserWarning):
        tdata_subset.obst["new_tree"] = new_tree
    assert not tdata_subset.is_view
    assert list(tdata_subset.obst.keys()) == ["tree", "new_tree"]
    assert list(tdata_subset.obst["new_tree"].edges) == [("0", "8")]
    # good assignment no overlap
    tree = nx.DiGraph([("root", "0"), ("root", "1")])
    good_tree = nx.DiGraph([("root", "2"), ("root", "3")])
    tdata = td.TreeData(X=np.zeros((8, 8)), allow_overlap=False, obst={"tree": tree})
    print(tdata.obs_names)
    tdata_subset = tdata[:4, :]
    print(tdata_subset.obs_names)
    with pytest.warns(UserWarning):
        tdata_subset.obst["tree"] = good_tree


def test_views_del(tdata):
    tdata_subset = tdata[[0, 1, 4], :]
    # bad deletion
    with pytest.raises(KeyError):
        del tdata_subset.obst["bad"]
    assert tdata_subset.is_view
    # good deletion actualizes object
    with pytest.warns(UserWarning):
        del tdata_subset.obst["tree"]
    assert not tdata_subset.is_view
    assert list(tdata_subset.obst.keys()) == []


def test_views_contains(tdata):
    tdata_subset = tdata[[0, 1, 4], :]
    assert "tree" in tdata_subset.obst
    assert "bad" not in tdata_subset.obst


def test_views_len(tdata):
    tdata_subset = tdata[[0, 1, 4], :]
    assert len(tdata_subset.obst) == 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
