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
    yield td.TreeData(
        X=np.zeros((8, 8)), obst={"0": tree}, vart={"0": tree}, obs=df, var=df, allow_overlap=True, label="tree"
    )


@pytest.fixture
def nodes_tdata(tree):
    df = pd.DataFrame({"anno": range(15)}, index=[str(i) for i in range(15)])
    yield td.TreeData(
        X=np.zeros((15, 15)),
        obst={"tree": tree},
        vart={"tree": tree},
        obs=df,
        var=df,
        alignment="nodes",
        allow_overlap=True,
    )


@pytest.fixture
def tdata_list(tdata):
    other_tree = nx.DiGraph()
    other_tree.add_edges_from([("0", "7"), ("0", "8")])
    tdata_1 = tdata[:2, :].copy()
    tdata_1.obst["1"] = other_tree
    tdata_1.vart["1"] = other_tree
    yield [tdata_1, tdata[2:4, :].copy(), tdata[4:, :4].copy()]


def test_concat(tdata_list):
    # outer join
    tdata = td.concat(tdata_list, axis="obs", label="subset", join="outer")
    assert list(tdata.obs["subset"]) == ["0"] * 2 + ["1"] * 2 + ["2"] * 4
    assert tdata.obst["0"].number_of_nodes() == 15
    assert tdata.obst["1"].number_of_nodes() == 3
    assert tdata.shape == (8, 8)
    # inner join
    tdata = td.concat(tdata_list, axis=0, label="subset", join="inner")
    assert list(tdata.obs["subset"]) == ["0"] * 2 + ["1"] * 2 + ["2"] * 4
    assert tdata.shape == (8, 4)
    # test join dictonary
    tdata_dict = {str(i): tdata for i, tdata in enumerate(tdata_list)}
    tdata = td.concat(tdata_dict, axis=0, label="subset", join="inner")
    assert tdata.shape == (8, 4)


def test_merge_outer(tdata_list):
    # None
    tdata = td.concat(tdata_list, axis=0, join="outer", merge=None)
    assert list(tdata.vart.keys()) == []
    # same
    tdata = td.concat(tdata_list, axis=0, join="outer", merge="same")
    assert list(tdata.vart.keys()) == []
    # unique
    tdata = td.concat(tdata_list, axis=0, join="outer", merge="first")
    assert list(tdata.vart.keys()) == ["0", "1"]
    # only
    tdata = td.concat(tdata_list, axis=0, join="outer", merge="only")
    assert list(tdata.vart.keys()) == ["1"]
    # first
    tdata = td.concat(tdata_list, axis=0, join="outer", merge="first")
    assert list(tdata.vart.keys()) == ["0", "1"]
    assert tdata.vart["0"].number_of_nodes() == 15
    assert tdata.vart["1"].number_of_nodes() == 3


def test_merge_inner(tdata_list):
    # None
    tdata = td.concat(tdata_list, axis=0, join="inner", merge=None)
    assert list(tdata.vart.keys()) == []
    # same
    tdata = td.concat(tdata_list, axis=0, join="inner", merge="same")
    assert list(tdata.vart.keys()) == ["0"]
    # unique
    tdata = td.concat(tdata_list, axis=0, join="inner", merge="first")
    assert list(tdata.vart.keys()) == ["0", "1"]
    # only
    tdata = td.concat(tdata_list, axis=0, join="inner", merge="only")
    assert list(tdata.vart.keys()) == ["1"]
    # first
    tdata = td.concat(tdata_list, axis=0, join="inner", merge="first")
    assert list(tdata.vart.keys()) == ["0", "1"]
    assert tdata.vart["0"].number_of_nodes() == 8
    assert tdata.vart["1"].number_of_nodes() == 3


def test_node_alignment(nodes_tdata):
    tdata1 = nodes_tdata[:14, :].copy()
    tdata2 = nodes_tdata[14:, :].copy()
    assert tdata1.alignment == "nodes"
    assert tdata2.alignment == "nodes"
    # concat same obst key
    with pytest.warns(UserWarning):
        tdata = td.concat([tdata1, tdata2], axis=0, join="outer", merge="same")
    assert tdata.shape == (15, 15)
    print(tdata)
    assert tdata.alignment == "nodes"
    assert list(tdata.obst.keys()) == []
    assert list(tdata.vart.keys()) == ["tree"]
    # concat different obst keys
    tdata2.obst["tree2"] = tdata2.obst.pop("tree")
    tdata = td.concat([tdata1, tdata2], axis=0, join="outer", merge="same")
    assert tdata.shape == (15, 15)
    assert tdata.alignment == "nodes"
    assert set(tdata.obst.keys()) == {"tree", "tree2"}
    assert list(tdata.vart.keys()) == ["tree"]
    assert tdata.obst["tree"].number_of_nodes() == 14
    assert tdata.obst["tree2"].number_of_nodes() == 1


def test_subset_alignment(nodes_tdata):
    with pytest.warns(UserWarning):
        tdata1 = nodes_tdata[[0, 2, 4, 6, 8, 10, 12, 14], :].copy()
    with pytest.warns(UserWarning):
        tdata2 = nodes_tdata[[1, 3, 5, 7, 9, 11, 13], :].copy()
    assert tdata1.alignment == "subset"
    assert tdata2.alignment == "subset"
    # concat same obst key
    with pytest.warns(UserWarning):
        tdata = td.concat([tdata1, tdata2], axis=0, join="outer", merge="same")
    assert tdata.shape == (15, 15)
    assert tdata.alignment == "subset"
    assert list(tdata.obst.keys()) == ["tree"]
    assert list(tdata.vart.keys()) == ["tree"]


def test_concat_bad_index(tdata_list):
    tdata_list[0].obs.index = tdata_list[1].obs.index
    with pytest.raises(ValueError):
        td.concat(tdata_list, axis=0, join="outer")


def test_concat_bad_tree(tdata_list):
    bad_tree = nx.DiGraph()
    bad_tree.add_edges_from([("bad", "7")])
    tdata_list[0].obst["0"] = bad_tree
    with pytest.raises(ValueError):
        td.concat(tdata_list, axis=0, join="outer")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
