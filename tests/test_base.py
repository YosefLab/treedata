import pytest

import anndata as ad
import networkx as nx
import numpy as np
import treedata as td

@pytest.fixture
def X():
    yield np.array([[1, 2, 3], 
                    [4, 5, 6],
                    [7, 8, 9]])

@pytest.fixture
def adata(X):
    yield ad.AnnData(X)

@pytest.fixture
def tree():
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "0"), ("root", "1")])
    yield tree

def test_package_has_version():
    assert td.__version__ is not None

def test_creation(X,adata,tree):
    # Test creation with np array
    tdata = td.TreeData(X,obst={"tree": tree},vart={"tree": tree},label = None)
    assert tdata.obst["tree"] == tree
    assert tdata.vart["tree"] == tree
    # Test creation with anndata
    tdata = td.TreeData(adata)
    assert tdata.X is adata.X

@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_keys(X,tree,dim):
    tdata = td.TreeData(X,obst={"tree": tree},vart={"tree": tree},label = None)
    assert getattr(tdata, f"{dim}t_keys")() == ["tree"]

@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_set(X,tree,dim):
    tdata = td.TreeData(X)
    setattr(tdata, f"{dim}t", {"tree": tree})
    assert getattr(tdata, f"{dim}t")["tree"] == tree

@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_del(X,tree,dim):
    tdata = td.TreeData(X,obst={"tree": tree},vart={"tree": tree},label = None)
    del getattr(tdata, f"{dim}t")["tree"]
    assert getattr(tdata, f"{dim}t") == {}

@pytest.mark.parametrize("dim", ["obs", "var"])
def test_tree_contains(X,tree,dim):
    tdata = td.TreeData(X,obst={"tree": tree},vart={"tree": tree},label = None)
    assert "tree" in getattr(tdata, f"{dim}t")
    assert "not_tree" not in getattr(tdata, f"{dim}t")

def test_tree_label(X,tree):
    # Test tree label
    second_tree = nx.DiGraph()
    second_tree.add_edges_from([("root", "2")])
    tdata = td.TreeData(X, obst={"0": tree, "1": second_tree},label = "tree")
    assert tdata.obs["tree"].tolist() == ["0", "0", "1"]
    # Test tree label with overlap
    tdata = td.TreeData(X, obst={"0": tree, "1": tree},label = "tree",
                        allow_tree_overlap = True)
    assert tdata.obs["tree"].tolist() == [["0","1"], ["0","1"], []]

def test_tree_overlap(X,tree):
    second_tree = nx.DiGraph()
    second_tree.add_edges_from([("root", "0"), ("root", "1")])
    # Test overlap not allowed
    with pytest.raises(ValueError):
        tdata = td.TreeData(X, obst={"0": tree, "1": second_tree},
                            allow_tree_overlap = False)
    # Test overlap allowed
    tdata = td.TreeData(X, obst={"0": tree, "1": second_tree},
                        allow_tree_overlap = True)
    assert tdata.obst == {"0": tree, "1": second_tree}

def test_repr(X, tree):
    tdata = td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None)
    expected_repr = f"TreeData object with n_obs × n_vars = {X.shape[0]} × {X.shape[1]}"
    expected_repr += f"\n    obst: 'tree'"
    expected_repr += f"\n    vart: 'tree'"
    assert repr(tdata) == expected_repr