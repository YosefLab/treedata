import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import treedata as td


@pytest.fixture
def X():
    yield np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def adata(X):
    yield ad.AnnData(X)


@pytest.fixture
def tree():
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "0"), ("root", "1")])
    yield tree


def check_graph_equality(g1, g2):
    assert nx.is_isomorphic(g1, g2, node_match=lambda n1, n2: n1 == n2, edge_match=lambda e1, e2: e1 == e2)


def test_creation(X, adata, tree):
    # Test creation with np array
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    print(type(tdata))
    check_graph_equality(tdata.obst["tree"], tree)
    check_graph_equality(tdata.vart["tree"], tree)
    # Test creation with anndata
    tdata = td.TreeData(adata)
    assert tdata.X is adata.X


@pytest.mark.parametrize("axis", [0, 1])
def test_attributes(X, tree, axis):
    dim = ["obs", "var"][axis]
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    assert getattr(tdata, f"{dim}t").axes == (axis,)
    assert getattr(tdata, f"{dim}t").attrname == (f"{dim}t")
    assert getattr(tdata, f"{dim}t").dim == dim
    assert getattr(tdata, f"{dim}t").parent is tdata
    assert list(getattr(tdata, f"{dim}t").dim_names) == ["0", "1", "2"]
    assert tdata.allow_overlap is False
    assert tdata.label is None


@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_set(X, tree, dim):
    tdata = td.TreeData(X)
    setattr(tdata, f"{dim}t", {"tree": tree})
    check_graph_equality(getattr(tdata, f"{dim}t")["tree"], tree)


@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_del(X, tree, dim):
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    del getattr(tdata, f"{dim}t")["tree"]
    assert getattr(tdata, f"{dim}t") == {}


@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_contains(X, tree, dim):
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    assert "tree" in getattr(tdata, f"{dim}t")
    assert "not_tree" not in getattr(tdata, f"{dim}t")


@pytest.mark.filterwarnings
@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_label(X, tree, dim):
    # Test tree label
    second_tree = nx.DiGraph()
    second_tree.add_edges_from([("root", "2")])
    tdata = td.TreeData(X, obst={"0": tree, "1": second_tree}, vart={"0": tree, "1": second_tree}, label="tree")
    assert getattr(tdata, dim)["tree"].tolist() == ["0", "0", "1"]
    # Test tree label with overlap
    tdata = td.TreeData(X, obst={"0": tree, "1": tree}, label="tree", vart={"0": tree, "1": tree}, allow_overlap=True)
    assert getattr(tdata, dim).loc["0", "tree"] == "0,1"
    # Test label already present warning
    df = pd.DataFrame({"tree": ["bad", "bad", "bad"]})
    with pytest.warns(UserWarning):
        tdata = td.TreeData(X, label="tree", obs=df, var=df)
    # Test tree label with updata
    tdata = td.TreeData(X, obst={"0": tree, "1": tree}, label="tree", vart={"0": tree, "1": tree}, allow_overlap=True)
    tdata.obst["0"] = tree
    assert getattr(tdata, dim).loc["0", "tree"] == "0,1"


def test_tree_overlap(X, tree):
    second_tree = nx.DiGraph()
    second_tree.add_edges_from([("root", "0"), ("root", "1")])
    # Test overlap not allowed
    with pytest.raises(ValueError):
        tdata = td.TreeData(X, obst={"0": tree, "1": second_tree}, allow_overlap=False)
    # Test overlap allowed
    tdata = td.TreeData(X, obst={"0": tree, "1": second_tree}, allow_overlap=True)
    check_graph_equality(tdata.obst["0"], tree)
    check_graph_equality(tdata.obst["1"], second_tree)


def test_repr(X, tree):
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    # TreeData
    expected_repr = f"TreeData object with n_obs × n_vars = {X.shape[0]} × {X.shape[1]}"
    expected_repr += "\n    obst: 'tree'"
    expected_repr += "\n    vart: 'tree'"
    assert repr(tdata) == expected_repr
    # AxisTrees
    expected_repr = "AxisTrees with keys: tree"
    assert repr(tdata.obst) == expected_repr


def test_mutability(X, tree):
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    # Toplogy is immutable
    with pytest.raises(nx.NetworkXError):
        tdata.obst["tree"].remove_node("0")
    # Attributes are mutable
    nx.set_node_attributes(tdata.obst["tree"], True, "test")
    assert all(tdata.obst["tree"].nodes[node]["test"] for node in tdata.obst["tree"].nodes)
    # Topology mutable on copy
    tree = tdata.obst["tree"].copy()
    tree.remove_node("1")
    tdata.obst["tree"] = tree
    assert list(tdata.obst["tree"].nodes) == ["root", "0"]


def test_bad_tree(X):
    # Not directed graph
    not_di_graph = nx.Graph()
    with pytest.raises(ValueError):
        _ = td.TreeData(X, obst={"tree": not_di_graph})
    # Has cycle
    has_cycle = nx.DiGraph()
    has_cycle.add_edges_from([("0", "1"), ("1", "0")])
    has_cycle.add_node("2")
    with pytest.raises(ValueError):
        _ = td.TreeData(X, obst={"tree": has_cycle})
    # Not fully connected
    not_fully_connected = nx.DiGraph()
    not_fully_connected.add_edges_from([("root", "0"), ("bad", "1")])
    with pytest.raises(ValueError):
        _ = td.TreeData(X, obst={"tree": not_fully_connected})
    # Leaves not in dim_names
    bad_leaves = nx.DiGraph()
    bad_leaves.add_edges_from([("root", "0"), ("root", "bad")])
    with pytest.raises(ValueError):
        _ = td.TreeData(X, obst={"tree": bad_leaves})
    # Multiple roots
    multi_root = nx.DiGraph()
    multi_root.add_edges_from([("0", "1"), ("1", "0"), ("2", "3")])
    with pytest.raises(ValueError):
        _ = td.TreeData(X, obst={"tree": multi_root})


def test_to_adata(X, tree):
    obs = pd.DataFrame({"cell": ["A", "B", "B"]}, index=["0", "1", "2"])
    tdata = td.TreeData(X, obst={"tree": tree}, obs=obs)
    adata = tdata.to_adata()
    assert type(adata) is ad.AnnData
    assert tdata.X is adata.X
    assert tdata.obs["cell"].tolist() == adata.obs["cell"].tolist()


def test_copy(adata, tree):
    treedata = td.TreeData(adata, obst={"tree": tree})
    treedata_copy = treedata.copy()
    assert np.array_equal(treedata.X, treedata_copy.X)
    assert treedata.obst["tree"].nodes == treedata_copy.obst["tree"].nodes
    assert treedata.obst["tree"].edges == treedata_copy.obst["tree"].edges


def test_transpose(adata, tree):
    treedata = td.TreeData(adata, obst={"tree": tree})
    treedata_transpose = treedata.transpose()
    assert np.array_equal(treedata.X.T, treedata_transpose.X)
    assert treedata.obst["tree"].nodes == treedata_transpose.vart["tree"].nodes
    assert treedata_transpose.obst_keys() == []
    assert np.array_equal(treedata.obs_names, treedata.T.obs_names)


@pytest.mark.parametrize("dim", ["obs", "var"])
def test_not_unique(X, tree, dim):
    with pytest.warns(UserWarning):
        tdata = td.TreeData(pd.DataFrame(X, index=["0", "1", "1"], columns=["0", "1", "1"]))
    assert not getattr(tdata, f"{dim}_names").is_unique
    with pytest.warns(UserWarning):
        setattr(tdata, f"{dim}t", {"tree": tree})
    assert getattr(tdata, f"{dim}_names").is_unique
