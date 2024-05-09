import joblib
import networkx as nx
import numpy as np
import pytest

import treedata as td


@pytest.fixture
def X():
    yield np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def tree():
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "0"), ("root", "1")])
    tree["root"]["0"]["length"] = 1
    tree.nodes["root"]["depth"] = 0
    yield tree


@pytest.fixture
def tdata(X, tree):
    yield td.TreeData(X, obst={"tree": tree}, vart={"tree": tree}, label=None, allow_overlap=False)


def check_graph_equality(g1, g2):
    assert nx.is_isomorphic(g1, g2, node_match=lambda n1, n2: n1 == n2, edge_match=lambda e1, e2: e1 == e2)


def test_h5ad_readwrite(tdata, tmp_path):
    # not backed
    file_path = tmp_path / "test.h5ad"
    tdata.write_h5ad(file_path)
    tdata2 = td.read_h5ad(file_path)
    assert np.array_equal(tdata2.X, tdata.X)
    check_graph_equality(tdata2.obst["tree"], tdata.obst["tree"])
    check_graph_equality(tdata2.vart["tree"], tdata.vart["tree"])
    assert tdata2.label is None
    assert tdata2.allow_overlap is False
    # backed
    tdata2 = td.read_h5ad(file_path, backed="r")
    assert np.array_equal(tdata2.X, tdata.X)
    check_graph_equality(tdata2.obst["tree"], tdata.obst["tree"])
    check_graph_equality(tdata2.vart["tree"], tdata.vart["tree"])
    assert tdata2.label is None
    assert tdata2.allow_overlap is False
    assert tdata2.isbacked
    assert tdata2.file.is_open
    assert tdata2.filename == file_path


def test_zarr_readwrite(tdata, tmp_path):
    tdata.write_zarr(tmp_path / "test.zarr")
    tdata2 = td.read_zarr(tmp_path / "test.zarr")
    assert np.array_equal(tdata2.X, tdata.X)
    check_graph_equality(tdata2.obst["tree"], tdata.obst["tree"])
    check_graph_equality(tdata2.vart["tree"], tdata.vart["tree"])
    assert tdata2.label is None
    assert tdata2.allow_overlap is False


def test_read_anndata(tdata, tmp_path):
    adata = tdata.to_adata()
    file_path = tmp_path / "test.h5ad"
    adata.write_h5ad(file_path)
    tdata = td.read_h5ad(file_path)
    assert np.array_equal(tdata.X, adata.X)
    assert tdata.label is None
    assert tdata.allow_overlap is False
    assert tdata.obst_keys() == []


def test_h5ad_backing(tdata, tree, tmp_path):
    tdata_copy = tdata.copy()
    assert not tdata.isbacked
    backing_h5ad = tmp_path / "test.h5ad"
    tdata.filename = backing_h5ad
    # backing mode
    tdata.write()
    assert not tdata.file.is_open
    assert tdata.isbacked
    # view of backed object
    tdata_subset = tdata[:, 0]
    subset_hash = joblib.hash(tdata_subset)
    assert tdata_subset.is_view
    assert tdata_subset.isbacked
    assert tdata_subset.shape == (3, 1)
    check_graph_equality(tdata_subset.obst["tree"], tdata.obst["tree"])
    assert np.array_equal(tdata_subset.X, tdata_copy.X[:, 0].reshape(-1, 1))
    # cannot set view in backing mode...
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            tdata_subset.obs["foo"] = range(3)
    # with pytest.warns(UserWarning):
    #    with pytest.raises(ValueError):
    #       tdata_subset.obst["foo"] = tree
    assert subset_hash == joblib.hash(tdata_subset)
    assert tdata_subset.is_view
    # copy
    tdata_subset = tdata_subset.copy(tmp_path / "test.subset.h5ad")
    assert not tdata_subset.is_view
    tdata_subset.obs["foo"] = range(3)
    assert not tdata_subset.is_view
    assert tdata_subset.isbacked
    assert tdata_subset.obs["foo"].tolist() == list(range(3))
    tdata_subset.write()
    # move to memory
    tdata_subset = tdata_subset.to_memory()
    assert not tdata_subset.is_view
    assert not tdata_subset.isbacked
    check_graph_equality(tdata_subset.obst["tree"], tdata.obst["tree"])
