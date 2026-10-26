import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import treedata as td


@pytest.fixture
def nwk(tmp_path):
    def _nwk(text, name="tree.nwk"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return _nwk


def leaves(tree):
    return [n for n in tree if tree.out_degree(n) == 0]


def test_io_module():
    assert td.io.read_nwk is not None
    assert td.io.read_h5td is td.read_h5td
    assert td.io.read_zarr is td.read_zarr
    assert not hasattr(td, "read_nwk")


def test_read_nwk_lengths(nwk):
    tdata = td.io.read_nwk(nwk("((A:1,B:2):3,(C:1,D:1):2);"))
    assert isinstance(tdata, td.TreeData)
    assert tdata.shape == (4, 0)
    assert list(tdata.obs_names) == ["A", "B", "C", "D"]
    assert list(tdata.obst) == ["tree"]
    tree = tdata.obst["tree"]
    assert nx.is_tree(tree)
    assert tree.number_of_nodes() == 7
    root = next(n for n in tree if tree.in_degree(n) == 0)
    assert tree.nodes[root]["time"] == 0
    # `length` is stored on the edges and `time` is the distance from the root on the nodes
    assert {(tree.nodes[p]["time"], tree.edges[p, c]["length"]) for p, c in tree.edges} == {
        (0, 3),
        (0, 2),
        (3, 1),
        (3, 2),
        (2, 1),
    }
    assert {n: tree.nodes[n]["time"] for n in leaves(tree)} == {"A": 4, "B": 5, "C": 3, "D": 3}
    for p, c in tree.edges:
        assert tree.nodes[c]["time"] == tree.nodes[p]["time"] + tree.edges[p, c]["length"]


def test_read_nwk_no_lengths(nwk):
    tdata = td.io.read_nwk(nwk("((A,B),(C,D));"))
    tree = tdata.obst["tree"]
    assert list(tdata.obs_names) == ["A", "B", "C", "D"]
    assert all("time" not in attrs for _, attrs in tree.nodes(data=True))
    assert all("length" not in attrs for _, _, attrs in tree.edges(data=True))


def test_read_nwk_root_length(nwk):
    tree = td.io.read_nwk(nwk("(A:1,B:2):0.5;")).obst["tree"]
    assert [tree.nodes[n]["time"] for n in tree if tree.in_degree(n) == 0] == [0.5]
    assert tree.nodes["A"]["time"] == 1.5
    assert tree.nodes["B"]["time"] == 2.5


def test_read_nwk_only_root_length(nwk):
    # A root length is a branch length, so a single node can have a `time`
    with warnings.catch_warnings():
        # Only this warning, as dependencies can emit unrelated ones, e.g. pandas deprecations
        warnings.filterwarnings("error", message="Some branches have no length")
        tree = td.io.read_nwk(nwk("A:1;")).obst["tree"]
    assert dict(tree.nodes(data=True)) == {"A": {"time": 1}}
    # The branches below a root with a length but without lengths of their own are missing
    with pytest.warns(UserWarning, match="no length"):
        tree = td.io.read_nwk(nwk("(A,B):0.5;")).obst["tree"]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    assert tree.nodes[root]["time"] == 0.5
    assert all(np.isnan(attrs["length"]) for _, _, attrs in tree.edges(data=True))
    assert all(np.isnan(tree.nodes[n]["time"]) for n in ["A", "B"])


def test_read_nwk_internal_labels(nwk):
    tree = td.io.read_nwk(nwk("((A:1,B:2)95:3,(C:1,D:1)clade:2)100;")).obst["tree"]
    # Numbers on internal nodes are support values and other labels are names
    assert "clade" in tree
    assert "95" not in tree and "100" not in tree
    assert sorted(attrs["support"] for _, attrs in tree.nodes(data=True) if "support" in attrs) == [95, 100]
    assert tree.nodes["A"].keys() == {"time"}
    # Numbers on leaves and quoted labels are names
    tree = td.io.read_nwk(nwk("((1,2)'3',4);")).obst["tree"]
    assert set(tree) >= {"1", "2", "3", "4"}
    assert not any("support" in attrs for _, attrs in tree.nodes(data=True))


def test_read_nwk_quoted_labels(nwk):
    tdata = td.io.read_nwk(nwk("('Homo sapiens':1,'it''s (a, b); [c]':2,C_3:0.5);"))
    assert list(tdata.obs_names) == ["Homo sapiens", "it's (a, b); [c]", "C_3"]


def test_read_nwk_comments(nwk):
    text = "[&R] ((A[&rate=0.1,height={1,2}]:1.5e-2,B:-1)[&&NHX:S=x[nested]]:2,C:3); [trailing]"
    tdata = td.io.read_nwk(nwk(text))
    tree = tdata.obst["tree"]
    assert list(tdata.obs_names) == ["A", "B", "C"]
    assert tree.nodes["A"]["time"] == pytest.approx(2.015)
    assert tree.nodes["B"]["time"] == 1


def test_read_nwk_whitespace_and_semicolon(nwk):
    expected = td.io.read_nwk(nwk("(A:1,B:2);"))
    for text in ["(\n  A : 1 ,\n  B : 2\n)\n;\n", "(A:1,B:2)", "﻿(A:1,B:2);"]:
        tdata = td.io.read_nwk(nwk(text))
        assert list(tdata.obs_names) == ["A", "B"]
        assert nx.utils.graphs_equal(tdata.obst["tree"], expected.obst["tree"])


def test_read_nwk_unnamed_nodes(nwk):
    tdata = td.io.read_nwk(nwk("(,(,));"))
    tree = tdata.obst["tree"]
    assert tdata.n_obs == 3
    assert tree.number_of_nodes() == 5
    assert set(tdata.obs_names) == set(leaves(tree))
    # Generated names do not clash with names in the file
    tree = td.io.read_nwk(nwk("(node0,(node1,));")).obst["tree"]
    assert tree.number_of_nodes() == 5
    assert {"node0", "node1"} <= set(tree)


def test_read_nwk_single_node(nwk):
    tdata = td.io.read_nwk(nwk("A;"))
    assert tdata.shape == (1, 0)
    assert list(tdata.obst["tree"]) == ["A"]


def test_read_nwk_multiple_trees(nwk):
    tdata = td.io.read_nwk(nwk("((A,B),C);\n((A,C),B);\n(D,E);"), key="nj")
    assert list(tdata.obst) == ["nj_0", "nj_1", "nj_2"]
    assert list(tdata.obs_names) == ["A", "B", "C", "D", "E"]
    # Generated names are not shared by trees
    assert set(tdata.obst["nj_0"]).isdisjoint(set(tdata.obst["nj_1"]) - {"A", "B", "C"})
    assert tdata.obs.loc["A", "tree"] == "nj_0,nj_1"
    assert tdata.obs.loc["D", "tree"] == "nj_2"


def test_read_nwk_partial_lengths(nwk):
    with pytest.warns(UserWarning, match="no length"):
        tree = td.io.read_nwk(nwk("((A:1,B),C:2);")).obst["tree"]
    assert np.isnan(tree.nodes["B"]["time"])
    assert tree.nodes["C"]["time"] == 2


def test_read_nwk_deep_tree(nwk):
    n = 5000
    text = "(" * n + "A:1" + "".join(f",B{i}:1):1" for i in range(n)).removesuffix(":1") + ";"
    tree = td.io.read_nwk(nwk(text)).obst["tree"]
    assert tree.number_of_nodes() == 2 * n + 1
    assert tree.nodes["A"]["time"] == n


def test_read_nwk_aligned(nwk):
    tdata = td.io.read_nwk(nwk("((A,B),C);"))
    assert tdata.alignment == "leaves"
    assert list(tdata.obs["tree"]) == ["tree"] * 3
    subset = tdata[["A", "C"]]
    assert set(leaves(subset.obst["tree"])) == {"A", "C"}


def test_read_nwk_alignment_nodes(nwk):
    path = nwk("((A:1,B:2)AB:3,C:2);")
    tdata = td.io.read_nwk(path, alignment="nodes")
    tree = tdata.obst["tree"]
    assert tdata.alignment == "nodes"
    assert tdata.shape == (5, 0)
    # All nodes are observations, including the unnamed root, in the order of the file
    assert list(tdata.obs_names) == ["node0", "AB", "A", "B", "C"]
    assert set(tdata.obs_names) == set(tree)
    assert list(tdata.obs["tree"]) == ["tree"] * 5
    assert tree.nodes["B"]["time"] == 5
    # Same tree as with the default alignment
    assert nx.utils.graphs_equal(tree, td.io.read_nwk(path).obst["tree"])
    subset = tdata[["node0", "AB", "A"]]
    assert set(subset.obst["tree"]) == {"node0", "AB", "A"}


def test_read_nwk_alignment_nodes_multiple_trees(nwk):
    tdata = td.io.read_nwk(nwk("((A,B)x,C);\n((A,C),B)x;"), alignment="nodes")
    # `x` is an internal node of the first tree and the root of the second one
    assert list(tdata.obs_names) == ["node0", "x", "A", "B", "C", "node1"]
    assert tdata.obs.loc["x", "tree"] == "tree_0,tree_1"
    assert tdata.obs.loc["node1", "tree"] == "tree_1"


def test_read_nwk_invalid_alignment(nwk):
    path = nwk("((A,B),C);")
    for alignment in ["subset", "root"]:
        with pytest.raises(ValueError, match="alignment"):
            td.io.read_nwk(path, alignment=alignment)


@pytest.mark.parametrize("alignment, n_obs", [("leaves", 4), ("nodes", 7)])
@pytest.mark.parametrize("suffix", ["h5td", "zarr"])
def test_read_nwk_roundtrip(nwk, tmp_path, suffix, alignment, n_obs):
    tdata = td.io.read_nwk(nwk("((A:1,B:2)95:3,(C:1,D:1)x:2);"), alignment=alignment)
    path = tmp_path / f"tree.{suffix}"
    getattr(tdata, f"write_{suffix}")(path)
    tdata2 = getattr(td, f"read_{suffix}")(path)
    assert tdata2.shape == (n_obs, 0)
    assert tdata2.alignment == alignment
    assert nx.utils.graphs_equal(tdata.obst["tree"], tdata2.obst["tree"])


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "No trees"),
        (" ;; ", "No trees"),
        ("#NEXUS\nbegin trees;", "NEXUS"),
        ("(A,B", "incomplete"),
        ("(A,B))", "unbalanced"),
        ("A,B;", "unexpected ','"),
        ("(A,B)(C,D);", "unexpected '\\('"),
        ("(A:1:2,B);", "unexpected ':'"),
        ("(A:x,B);", "invalid branch length"),
        ("(A:,B);", "expected a branch length"),
        ("(A:1,B:);", "expected a branch length"),
        ("(A,B)'C'D;", "unexpected label"),
        ("(A:1[c]B,C);", "unexpected label"),
        ("(A:1B,C);", "invalid branch length"),
        ("('A,B);", "unterminated quoted"),
        ("(A[note,B);", "unterminated comment"),
        ("(A,B;(C,D);", "unexpected ';'"),
    ],
)
def test_read_nwk_invalid(nwk, text, match):
    with pytest.raises(ValueError, match=match):
        td.io.read_nwk(nwk(text))


def test_read_nwk_duplicate_names(nwk):
    with pytest.raises(ValueError, match="unique"):
        td.io.read_nwk(nwk("((A,B)C,(A,D)C);"))


@pytest.mark.parametrize(
    "text",
    [
        "((A:1,B:2)95:3,(C:1,D:1)x:2);",
        "((A,B)95,(C,D)x);",
        "(A:1,B:2):0.5;",
        "('a b':1,'it''s (x);':2,(C:1,D:1)'12':1);",
        "A:1;",
        "A;",
        "((A:1e-05,B:-2)x:3,C:2);",
        "((A,B),C);\n((A,C),B)x;",
    ],
)
@pytest.mark.parametrize("alignment", ["leaves", "nodes"])
def test_write_nwk_roundtrip(nwk, tmp_path, text, alignment):
    tdata = td.io.read_nwk(nwk(text), alignment=alignment)
    path = tmp_path / "out.nwk"
    td.io.write_nwk(path, tdata)
    tdata2 = td.io.read_nwk(path, alignment=alignment)
    assert list(tdata2.obs_names) == list(tdata.obs_names)
    assert list(tdata2.obst) == list(tdata.obst)
    for k in tdata.obst:
        assert nx.utils.graphs_equal(tdata.obst[k], tdata2.obst[k])


def test_write_nwk_text(tmp_path):
    tree = nx.DiGraph()
    tree.add_edge("root", "x", length=1.5)
    tree.add_edge("root", "y z", length=2)
    tree.add_edge("x", "A", length=1)
    tree.add_edge("x", "B", length=float("nan"))
    tree.nodes["x"]["support"] = 90
    tree.nodes["root"]["time"] = 0.25
    tdata = td.TreeData(obs=pd.DataFrame(index=["A", "B", "y z"]), obst={"tree": tree})
    tdata.write_nwk(tmp_path / "out.nwk")
    assert (tmp_path / "out.nwk").read_text() == "((A:1.0,B)90.0:1.5,'y z':2.0)root:0.25;\n"


def test_write_nwk_key(nwk, tmp_path):
    tdata = td.io.read_nwk(nwk("((A,B),C);\n(D,E);"))
    tdata.write_nwk(tmp_path / "all.nwk")
    assert len((tmp_path / "all.nwk").read_text().splitlines()) == 2
    tdata.write_nwk(tmp_path / "one.nwk", key="tree_1")
    tdata2 = td.io.read_nwk(tmp_path / "one.nwk")
    assert list(tdata2.obs_names) == ["D", "E"]
    with pytest.raises(KeyError, match="missing"):
        tdata.write_nwk(tmp_path / "x.nwk", key="missing")


def test_write_nwk_no_trees(tmp_path):
    with pytest.raises(ValueError, match="no non-empty trees"):
        td.TreeData(np.zeros((2, 1))).write_nwk(tmp_path / "x.nwk")


def test_write_nwk_deep_tree(nwk, tmp_path):
    n = 5000
    text = "(" * n + "A:1" + "".join(f",B{i}:1):1" for i in range(n)).removesuffix(":1") + ";"
    tdata = td.io.read_nwk(nwk(text))
    tdata.write_nwk(tmp_path / "out.nwk")
    assert nx.utils.graphs_equal(tdata.obst["tree"], td.io.read_nwk(tmp_path / "out.nwk").obst["tree"])


def test_write_nwk_view_with_empty_tree(nwk, tmp_path):
    tdata = td.io.read_nwk(nwk("((A,B),C);\n(D,E);"))
    view = tdata[["D", "E"]]
    assert view.is_view
    assert view.obst["tree_0"].number_of_nodes() == 0
    view.write_nwk(tmp_path / "view.nwk")
    tdata2 = td.io.read_nwk(tmp_path / "view.nwk")
    assert list(tdata2.obs_names) == ["D", "E"]
    # Nothing to write for a tree of which all observations were removed, and no file is left behind
    with pytest.raises(ValueError, match="no non-empty trees"):
        view.write_nwk(tmp_path / "empty.nwk", key="tree_0")
    assert not (tmp_path / "empty.nwk").exists()
