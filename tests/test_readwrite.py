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
def test_h5td_readwrite(tdata, tmp_path, backed):
    tdata.raw = tdata
    file_path = tmp_path / "test.h5td"
    tdata.write_h5td(file_path)
    tdata2 = td.read_h5td(file_path, backed=backed)
    assert np.array_equal(tdata2.X, tdata.X)
    assert np.array_equal(tdata2.raw.X, tdata.raw.X)
    check_graph_equality(tdata2.obst["1"], tdata.obst["1"])
    check_graph_equality(tdata2.vart["1"], tdata.vart["1"])
    assert tdata2.label == "tree"
    assert tdata2.allow_overlap is True
    assert tdata2.alignment == "leaves"
    assert tdata2.obst["1"].nodes["root"]["depth"] == 0
    assert tdata2.obst["2"].nodes["root"]["characters"] == ["-1", "1"]
    assert tdata2.obs.loc["0", "tree"] == "1,2"
    if backed:
        assert tdata2.isbacked
        assert tdata2.file.is_open
        assert tdata2.filename == file_path


def test_h5td_dtypes(tdata, tmp_path):
    file_path = tmp_path / "test.h5td"
    tdata.obst["1"].nodes["root"]["list"] = [1, 2, 3]
    tdata.obst["1"].nodes["root"]["tuple"] = (1, 2, 3)
    tdata.obst["1"].nodes["root"]["set"] = {1, 2, 3}
    tdata.obst["1"].nodes["root"]["np_float"] = np.float64(1.0)
    tdata.obst["1"].nodes["root"]["np_array"] = np.array([[1, 2], [3, 4]])
    tdata.obst["1"].nodes["root"]["pd_series"] = pd.Series(["1", "2", "3"])
    tdata.write_h5td(file_path)
    tdata2 = td.read_h5td(file_path)
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
    assert tdata2.alignment == "leaves"
    assert tdata2.obst["2"].nodes["root"]["depth"] == 0
    assert tdata2.obs.loc["0", "tree"] == "1,2"


def test_zarr_dtypes(tdata, tmp_path):
    """Columnar zarr storage round-trips all supported attribute types."""
    tdata.obst["1"].nodes["root"]["list"] = [1, 2, 3]
    tdata.obst["1"].nodes["root"]["tuple"] = (1, 2, 3)
    tdata.obst["1"].nodes["root"]["set"] = {1, 2, 3}
    tdata.obst["1"].nodes["root"]["np_float"] = np.float64(1.0)
    tdata.obst["1"].nodes["root"]["np_array"] = np.array([[1, 2], [3, 4]])
    tdata.obst["1"].nodes["root"]["pd_series"] = pd.Series(["a", "b", "c"])
    tdata.obst["1"].nodes["root"]["nested_list"] = [[1, 2], [3, 4], [5, 6]]
    tdata.obst["1"].nodes["root"]["str_val"] = "hello"
    tdata.obst["1"].nodes["root"]["int_val"] = 42
    tdata.obst["1"].nodes["root"]["float_val"] = 3.14
    tdata.obst["1"].nodes["root"]["bool_val"] = True
    tdata.write_zarr(tmp_path / "test.zarr")
    tdata2 = td.read_zarr(tmp_path / "test.zarr")
    root = tdata2.obst["1"].nodes["root"]
    assert root["list"] == [1, 2, 3] and isinstance(root["list"], list)
    assert root["tuple"] == [1, 2, 3] and isinstance(root["tuple"], list)
    assert root["set"] == [1, 2, 3] and isinstance(root["set"], list)
    assert root["np_float"] == 1.0 and isinstance(root["np_float"], float)
    assert root["np_array"] == [[1, 2], [3, 4]] and isinstance(root["np_array"], list)
    assert root["pd_series"] == ["a", "b", "c"] and isinstance(root["pd_series"], list)
    assert root["nested_list"] == [[1, 2], [3, 4], [5, 6]]
    assert root["str_val"] == "hello"
    assert root["int_val"] == 42
    assert root["float_val"] == pytest.approx(3.14)
    assert root["bool_val"] is True


def test_zarr_edge_attrs(tmp_path):
    """Edge attributes with complex types survive zarr round-trip."""
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "A"), ("root", "B"), ("A", "C")])
    tree["root"]["A"]["length"] = 1.5
    tree["root"]["A"]["tags"] = ["x", "y"]
    tree["root"]["B"]["length"] = 2.0
    tree["root"]["B"]["tags"] = ["z"]
    tree["A"]["C"]["length"] = 0.5
    tree["A"]["C"]["tags"] = []
    tdata = td.TreeData(np.eye(3), obst={"t": tree}, alignment="subset")
    tdata.write_zarr(tmp_path / "test.zarr")
    tdata2 = td.read_zarr(tmp_path / "test.zarr")
    G = tdata2.obst["t"]
    assert G["root"]["A"]["length"] == pytest.approx(1.5)
    assert G["root"]["A"]["tags"] == ["x", "y"]
    assert G["root"]["B"]["length"] == pytest.approx(2.0)
    assert G["root"]["B"]["tags"] == ["z"]
    assert G["A"]["C"]["tags"] == []


def test_zarr_missing_node_attrs(tmp_path):
    """Nodes without a particular attribute don't gain it as None after round-trip."""
    tree = nx.DiGraph()
    tree.add_nodes_from(["root", "A", "B"])
    tree.add_edges_from([("root", "A"), ("root", "B")])
    tree.nodes["root"]["depth"] = 0
    tree.nodes["A"]["depth"] = 1
    tree.nodes["A"]["label"] = "leaf_a"
    # B has no attributes at all
    tdata = td.TreeData(np.eye(3), obst={"t": tree}, alignment="subset")
    tdata.write_zarr(tmp_path / "test.zarr")
    tdata2 = td.read_zarr(tmp_path / "test.zarr")
    G = tdata2.obst["t"]
    assert G.nodes["root"]["depth"] == 0
    assert "label" not in G.nodes["root"]
    assert G.nodes["A"]["depth"] == 1
    assert G.nodes["A"]["label"] == "leaf_a"
    assert dict(G.nodes["B"]) == {}


def test_zarr_zip_store(tdata, tmp_path):
    """Writing to a ZipStore (the original bug report) works end-to-end."""
    import zarr

    zip_path = tmp_path / "test.zarr.zip"
    store = zarr.storage.ZipStore(str(zip_path), mode="w")
    tdata.write_zarr(store=store)
    store.close()
    tdata2 = td.read_zarr(str(zip_path))
    assert np.array_equal(tdata2.X, tdata.X)
    check_graph_equality(tdata2.obst["1"], tdata.obst["1"])
    assert tdata2.obst["1"].nodes["root"]["depth"] == 0
    assert tdata2.obst["1"].nodes["root"]["characters"] == ["-1", "1"]


def test_zarr_large_tree(tmp_path):
    """A tree with many nodes and multiple attributes per node round-trips correctly."""
    # balanced_tree(2, 10) has 2047 nodes — large enough to exercise the code path
    # that failed with scalar string storage, without being slow
    tree = nx.balanced_tree(2, 10, create_using=nx.DiGraph)
    for node in tree.nodes():
        tree.nodes[node]["tags"] = [str(node), f"node_{node}"]
        tree.nodes[node]["weight"] = float(node) / tree.number_of_nodes()
    for u, v in tree.edges():
        tree[u][v]["dist"] = abs(u - v)
        tree[u][v]["meta"] = [u, v]
    tdata = td.TreeData(np.eye(3), obst={"big": tree}, alignment="subset")
    tdata.write_zarr(tmp_path / "big.zarr")
    tdata2 = td.read_zarr(tmp_path / "big.zarr")
    G = tdata2.obst["big"]
    assert G.number_of_nodes() == tree.number_of_nodes()
    assert G.number_of_edges() == tree.number_of_edges()
    assert isinstance(G.nodes["0"]["tags"], list)
    assert G.nodes["0"]["tags"] == ["0", "node_0"]
    assert G["0"]["1"]["meta"] == [0, 1]


def test_read_anndata(X, tmp_path):
    adata = ad.AnnData(X)
    file_path = tmp_path / "test.h5ad"
    adata.write_h5ad(file_path)
    tdata = td.read_h5td(file_path)
    assert np.array_equal(tdata.X, adata.X)
    assert tdata.label == "tree"
    assert tdata.allow_overlap is True
    assert tdata.alignment == "leaves"
    assert list(tdata.obst.keys()) == []


def test_read_no_X(X, tmp_path):
    tdata = td.TreeData(obs=pd.DataFrame(index=["0", "1", "2"]))
    file_path = tmp_path / "test.h5td"
    tdata.write_h5td(file_path)
    tdata2 = td.read_h5td(file_path)
    assert tdata2.X is None


def deprecated_read_write(tdata, tmp_path):
    # Test deprecated read/write methods
    file_path = tmp_path / "test_deprecated.h5td"
    with pytest.warns(DeprecationWarning):
        tdata.write_h5ad(file_path)
    with pytest.warns(DeprecationWarning):
        td.read_h5ad(file_path)


def test_h5td_backing(tdata, tree, tmp_path):
    tdata_copy = tdata.copy()
    assert not tdata.isbacked
    backing_h5td = tmp_path / "test_backed.h5td"
    tdata.filename = backing_h5td
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
    tdata_subset = tdata_subset.copy(tmp_path / "test_subset.h5td")
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
