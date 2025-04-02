import anndata as ad
import joblib
import networkx as nx
import numpy as np
import pandas as pd
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
    tree.nodes["root"]["characters"] = ["-1", "1"]
    yield tree


@pytest.fixture
def tdata(X, tree):
    yield td.TreeData(X, obst={"1": tree, "2": tree}, vart={"1": tree}, label="tree", allow_overlap=True)


def check_graph_equality(g1, g2):
    assert nx.is_isomorphic(
        g1, g2, node_match=lambda n1, n2: set(n1) == set(n2), edge_match=lambda e1, e2: set(e1) == set(e2)
    )


@pytest.mark.parametrize("backed", [None, "r"])
def test_h5ad_readwrite(tdata, tmp_path, backed):
    tdata.raw = tdata
    file_path = tmp_path / "test.h5ad"
    tdata.write_h5ad(file_path)
    tdata2 = td.read_h5ad(file_path, backed=backed)
    assert np.array_equal(tdata2.X, tdata.X)
    assert np.array_equal(tdata2.raw.X, tdata.raw.X)
    check_graph_equality(tdata2.obst["1"], tdata.obst["1"])
    check_graph_equality(tdata2.vart["1"], tdata.vart["1"])
    assert tdata2.label == "tree"
    assert tdata2.allow_overlap is True
    assert tdata2.obst["1"].nodes["root"]["depth"] == 0
    assert tdata2.obst["2"].nodes["root"]["characters"] == ["-1", "1"]
    assert tdata2.obs.loc["0", "tree"] == "1,2"
    if backed:
        assert tdata2.isbacked
        assert tdata2.file.is_open
        assert tdata2.filename == file_path


def test_h5ad_dtypes(tdata, tmp_path):
    file_path = tmp_path / "test.h5ad"
    tdata.obst["1"].nodes["root"]["list"] = [1, 2, 3]
    tdata.obst["1"].nodes["root"]["tuple"] = (1, 2, 3)
    tdata.obst["1"].nodes["root"]["set"] = {1, 2, 3}
    tdata.obst["1"].nodes["root"]["np_float"] = np.float64(1.0)
    tdata.obst["1"].nodes["root"]["np_array"] = np.array([[1, 2], [3, 4]])
    tdata.obst["1"].nodes["root"]["pd_series"] = pd.Series(["1", "2", "3"])
    tdata.write_h5ad(file_path)
    tdata2 = td.read_h5ad(file_path)
    assert tdata2.obst["1"].nodes["root"]["list"] == [1, 2, 3]
    assert isinstance(tdata2.obst["1"].nodes["root"]["list"], list)
    assert tdata2.obst["1"].nodes["root"]["tuple"] == [1, 2, 3]
    assert isinstance(tdata2.obst["1"].nodes["root"]["tuple"], list)
    assert tdata2.obst["1"].nodes["root"]["set"] == [1, 2, 3]
    assert isinstance(tdata2.obst["1"].nodes["root"]["set"], list)
    assert tdata2.obst["1"].nodes["root"]["np_float"] == 1.0
    assert isinstance(tdata2.obst["1"].nodes["root"]["np_float"], float)
    assert tdata2.obst["1"].nodes["root"]["np_array"] == [[1, 2], [3, 4]]
    assert isinstance(tdata2.obst["1"].nodes["root"]["np_array"], list)
    assert tdata2.obst["1"].nodes["root"]["pd_series"] == ["1", "2", "3"]
    assert isinstance(tdata2.obst["1"].nodes["root"]["pd_series"], list)


def test_zarr_readwrite(tdata, tmp_path):
    tdata.write_zarr(tmp_path / "test.zarr")
    tdata2 = td.read_zarr(tmp_path / "test.zarr")
    assert np.array_equal(tdata2.X, tdata.X)
    check_graph_equality(tdata2.obst["1"], tdata.obst["1"])
    check_graph_equality(tdata2.vart["1"], tdata.vart["1"])
    assert tdata2.label == "tree"
    assert tdata2.allow_overlap is True
    assert tdata2.obst["2"].nodes["root"]["depth"] == 0
    assert tdata2.obs.loc["0", "tree"] == "1,2"


def test_read_anndata(X, tmp_path):
    adata = ad.AnnData(X)
    file_path = tmp_path / "test.h5ad"
    adata.write_h5ad(file_path)
    tdata = td.read_h5ad(file_path)
    assert np.array_equal(tdata.X, adata.X)
    assert tdata.label == "tree"
    assert tdata.allow_overlap is False
    assert tdata.obst_keys() == []


def test_read_no_X(X, tmp_path):
    tdata = td.TreeData(obs=pd.DataFrame(index=["0", "1", "2"]))
    file_path = tmp_path / "test.h5ad"
    tdata.write_h5ad(file_path)
    tdata2 = td.read_h5ad(file_path)
    assert tdata2.X is None


def test_h5ad_backing(tdata, tree, tmp_path):
    tdata_copy = tdata.copy()
    assert not tdata.isbacked
    backing_h5ad = tmp_path / "test_backed.h5ad"
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
    check_graph_equality(tdata_subset.obst["1"], tdata.obst["1"])
    assert np.array_equal(tdata_subset.X, tdata_copy.X[:, 0].reshape(-1, 1))
    # cannot set view in backing mode...
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            tdata_subset.obs["foo"] = range(3)
    assert subset_hash == joblib.hash(tdata_subset)
    assert tdata_subset.is_view
    # copy
    tdata_subset = tdata_subset.copy(tmp_path / "test_subset.h5ad")
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
    print(tdata_subset)
    check_graph_equality(tdata_subset.obst["1"], tdata.obst["1"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
