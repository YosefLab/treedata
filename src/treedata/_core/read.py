from __future__ import annotations

import itertools
import json
import re
import warnings
from collections import Counter
from collections.abc import Iterator, MutableMapping
from os import PathLike
from typing import Literal

import anndata as ad
import h5py
import networkx as nx
import pandas as pd
import zarr

from treedata._core.treedata import TreeData

# Sentinel distinguishing an absent attribute from an attribute explicitly set to ``None``.
_MISSING = object()


def _read_elem(elem):
    """Read an element from a store."""
    return ad.io.read_elem(elem)


def _dict_to_digraph(graph_dict: dict) -> nx.DiGraph:
    """Convert a dictionary to a networkx.DiGraph."""
    G = nx.DiGraph()
    # Add nodes and their attributes
    for node, attrs in graph_dict["nodes"].items():
        G.add_node(node, **attrs)
    # Add edges and their attributes
    for source, targets in graph_dict["edges"].items():
        for target, attrs in targets.items():
            G.add_edge(source, target, **attrs)
    return G


def _parse_axis_trees(data: str) -> dict:
    """Parse AxisTrees from a JSON string (HDF5 format)."""
    return {k: _dict_to_digraph(v) for k, v in json.loads(data).items()}


def _decode_attr_columns(cols: dict) -> dict[str, list]:
    """Decode columnar attribute arrays into per-element Python lists.

    Native numeric/bool arrays are returned as-is (every element present).
    Object arrays hold per-element JSON strings, where the empty string ``""``
    marks an absent attribute; it is decoded to the :data:`_MISSING` sentinel so
    that an attribute explicitly set to ``None`` (JSON ``"null"``) is preserved
    and kept distinct from a missing attribute.
    """
    decoded: dict[str, list] = {}
    for k, arr in cols.items():
        dtype = getattr(arr, "dtype", None)
        if dtype is not None and dtype.kind in "iufb":
            decoded[k] = list(arr.tolist())
        else:
            decoded[k] = [_MISSING if s == "" else json.loads(s) for s in arr]
    return decoded


def _read_axis_trees_zarr(g: zarr.Group) -> dict:
    """Read AxisTrees from a zarr group written in the columnar format."""
    trees = {}
    for name in g.keys():
        tg = g[name]
        if not isinstance(tg, zarr.Group):
            continue
        G = nx.DiGraph()
        nodes = list(_read_elem(tg["nodes"]))
        node_attrs = _decode_attr_columns(_read_elem(tg["node_attrs"])) if "node_attrs" in tg else {}
        for i, node in enumerate(nodes):
            G.add_node(node, **{k: col[i] for k, col in node_attrs.items() if col[i] is not _MISSING})
        edges = _read_elem(tg["edges"])
        edge_attrs = _decode_attr_columns(_read_elem(tg["edge_attrs"])) if "edge_attrs" in tg else {}
        for i, (src, dst) in enumerate(edges):
            G.add_edge(src, dst, **{k: col[i] for k, col in edge_attrs.items() if col[i] is not _MISSING})
        trees[name] = G
    return trees


def _parse_legacy(treedata_attrs: dict) -> dict:
    """Parse tree attributes from AnnData uns field."""
    if treedata_attrs is not None:
        for j in ["obst", "vart"]:
            if j in treedata_attrs:
                treedata_attrs[j] = {k: _dict_to_digraph(v) for k, v in treedata_attrs[j].items()}
        treedata_attrs["allow_overlap"] = bool(treedata_attrs["allow_overlap"])
        treedata_attrs["label"] = treedata_attrs["label"] if "label" in treedata_attrs.keys() else None
    return treedata_attrs


def _read_raw(f, backed):
    """Read raw from file."""
    d = {}
    for k in ["obs", "var"]:
        if f"raw/{k}" in f:
            d[k] = _read_elem(f[f"raw/{k}"])
    if not backed:
        d["X"] = _read_elem(f["raw/X"])
    return d


def _read_tree_elems(f) -> dict:
    """Read the elements TreeData adds to AnnData from group ``f``."""
    d = {}
    for k in ["label", "allow_overlap", "alignment"]:
        if k in f:
            d[k] = _read_elem(f[k])
    for k in ["obst", "vart"]:
        if k in f:
            elem = f[k]
            if isinstance(elem, zarr.Group):
                d[k] = _read_axis_trees_zarr(elem)
            else:
                d[k] = _parse_axis_trees(_read_elem(elem))
    return d


def read_tree_elems(group: h5py.Group | zarr.Group, adata: ad.AnnData) -> TreeData:
    """Build a TreeData from an AnnData and the tree elements stored in ``group``.

    The counterpart of :func:`treedata.write_tree_elems`. ``adata`` is the AnnData read
    from ``group`` (e.g. by mudata) and ``group`` holds the ``obst``, ``vart``, ``label``,
    ``allow_overlap`` and ``alignment`` elements next to it.

    Parameters
    ----------
    group
        HDF5 or Zarr group containing the tree elements.
    adata
        The AnnData read from ``group``.
    """
    return TreeData(adata, **_read_tree_elems(group))


def _read_tdata(f, filename, backed) -> dict:
    """Read TreeData from file."""
    d = {}
    if backed is None:
        backed = False
    elif backed is True:
        backed = "r"
    # Read X if not backed
    if not backed:
        if "X" in f:
            d["X"] = _read_elem(f["X"])
    else:
        d.update({"filename": filename, "filemode": backed})
    # Read standard elements
    for k in ["obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns"]:
        if k in f:
            d[k] = _read_elem(f[k])
    # Read raw
    if "raw" in f:
        d["raw"] = _read_raw(f, backed)
    d.update(_read_tree_elems(f))
    # Read legacy treedata format
    if "raw.treedata" in f:
        d.update(_parse_legacy(json.loads(_read_elem(f["raw.treedata"]))))
    return d


def read_h5td(
    filename: str | PathLike | None = None,
    backed: Literal["r", "r+"] | bool | None = None,
) -> TreeData:
    """Read `.h5td` or `.h5ad`-formatted hdf5 file.

    Parameters
    ----------
    filename
        File name of data file.
    backed
        If `'r'`, load :class:`~TreeData` in `backed` mode
        instead of fully loading it into memory (`memory` mode).
        If you want to modify backed attributes of the TreeData object,
        you need to choose `'r+'`.
    """
    with h5py.File(filename, "r") as f:
        d = _read_tdata(f, filename, backed)
    return TreeData(**d)


def read_h5ad(
    filename: str | PathLike | None = None,
    backed: Literal["r", "r+"] | bool | None = None,
) -> TreeData:
    """Read `.h5td` or `.h5ad`-formatted hdf5 file. Deprecated, use `read_h5td` instead.

    Parameters
    ----------
    filename
        File name of data file.
    backed
        If `'r'`, load :class:`~TreeData` in `backed` mode
        instead of fully loading it into memory (`memory` mode).
        If you want to modify backed attributes of the TreeData object,
        you need to choose `'r+'`.
    """
    warnings.warn(
        "read_h5ad has been renamed to read_h5td. read_h5ad will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_h5td(filename, backed=backed)


def _open_zarr_group(store, *, mode: str = "r") -> tuple[zarr.Group, bool]:
    """Open a Zarr group and signal whether it should be closed."""
    if isinstance(store, zarr.Group):
        return store, False
    # zarr v3 does not auto-detect zip files from string paths; open explicitly
    if isinstance(store, (str, PathLike)) and str(store).endswith(".zip"):
        store = zarr.storage.ZipStore(store, mode=mode)
    return zarr.open(store, mode=mode), True


def _close_zarr_group(group: zarr.Group) -> None:
    """Close the underlying store for an opened Zarr group if needed."""
    store = getattr(group, "store", None)
    if store is not None:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def read_zarr(store: str | PathLike | MutableMapping | zarr.Group) -> TreeData:
    """Read from a hierarchical Zarr array store.

    Parameters
    ----------
    store
        The filename, a :class:`~typing.MutableMapping`, or a Zarr storage class.
    """
    group, should_close = _open_zarr_group(store)
    d = _read_tdata(group, store, backed=False)
    if should_close:
        _close_zarr_group(group)
    return TreeData(**d)


_NWK_WHITESPACE = re.compile(r"\s*")
# A token is one of: punctuation, the start of a comment, a quoted label, a stray quote
# (an unterminated quoted label), or an unquoted label/length. The order matters.
_NWK_TOKEN = re.compile(
    r"""(?P<punct>[(),:;])|(?P<comment>\[)|(?P<quoted>'(?:[^']|'')*')|(?P<badquote>')|(?P<label>[^(),:;\[]+)"""
)
_NWK_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class _NwkNode:
    """A node of a parsed Newick tree."""

    __slots__ = ("children", "label", "length", "name", "quoted", "support", "time")

    def __init__(self):
        self.children: list[_NwkNode] = []
        self.label: str | None = None
        self.quoted = False
        self.length: float | None = None
        self.name: str | None = None
        self.support: float | None = None
        self.time: float = 0.0


def _nwk_tokens(text: str) -> Iterator[tuple[str, str, int]]:
    """Split Newick text into ``(kind, value, position)`` tokens, skipping whitespace and comments."""
    pos, end = 0, len(text)
    while True:
        pos = _NWK_WHITESPACE.match(text, pos).end()
        if pos >= end:
            return
        m = _NWK_TOKEN.match(text, pos)
        kind = m.lastgroup
        if kind == "comment":
            # Comments (e.g. NHX or BEAST annotations) may be nested
            depth = 0
            for i in range(pos, end):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        break
            else:
                raise ValueError(f"Invalid Newick: unterminated comment starting at position {pos}.")
            pos = i + 1
            continue
        if kind == "badquote":
            raise ValueError(f"Invalid Newick: unterminated quoted label starting at position {pos}.")
        value = m.group()
        if kind == "quoted":
            value = value[1:-1].replace("''", "'")
        elif kind == "label":
            value = value.rstrip()
        yield kind, value, pos
        pos = m.end()


def _parse_newick(text: str) -> list[_NwkNode]:
    """Parse Newick text into a list of root nodes, one per tree."""
    if text.lstrip().upper().startswith("#NEXUS"):
        raise ValueError("NEXUS files are not supported, expected a Newick file.")
    trees: list[_NwkNode] = []
    root = cur = None  # `cur` is the node currently being read
    stack: list[_NwkNode] = []  # open internal nodes
    expect_length = False
    for kind, value, pos in _nwk_tokens(text):
        if kind == "punct" and value == ";":
            if root is None:
                continue
            if stack or expect_length:
                raise ValueError(f"Invalid Newick: unexpected ';' at position {pos}.")
            trees.append(root)
            root = cur = None
            continue
        if root is None:
            root = cur = _NwkNode()
        if expect_length:
            if kind not in ("label", "quoted"):
                raise ValueError(f"Invalid Newick: expected a branch length after ':' at position {pos}.")
            try:
                cur.length = float(value)
            except ValueError:
                raise ValueError(f"Invalid Newick: invalid branch length {value!r} at position {pos}.") from None
            expect_length = False
        elif kind in ("label", "quoted"):
            if cur.label is not None or cur.length is not None:
                raise ValueError(f"Invalid Newick: unexpected label {value!r} at position {pos}.")
            cur.label, cur.quoted = value, kind == "quoted"
        elif value == "(":
            if cur.label is not None or cur.length is not None or cur.children:
                raise ValueError(f"Invalid Newick: unexpected '(' at position {pos}.")
            stack.append(cur)
            cur = _NwkNode()
            stack[-1].children.append(cur)
        elif value == ",":
            if not stack:
                raise ValueError(f"Invalid Newick: unexpected ',' at position {pos}.")
            cur = _NwkNode()
            stack[-1].children.append(cur)
        elif value == ")":
            if not stack:
                raise ValueError(f"Invalid Newick: unbalanced ')' at position {pos}.")
            cur = stack.pop()
        elif value == ":":
            if cur.length is not None:
                raise ValueError(f"Invalid Newick: unexpected ':' at position {pos}.")
            expect_length = True
    if root is not None:
        # The semicolon after the last tree is optional
        if stack or expect_length:
            raise ValueError("Invalid Newick: the tree is incomplete, check for unbalanced parentheses.")
        trees.append(root)
    if not trees:
        raise ValueError("No trees found in the Newick file.")
    return trees


def _preorder(root: _NwkNode) -> list[tuple[_NwkNode, _NwkNode | None]]:
    """List ``(node, parent)`` pairs in preorder, so that parents come before their children."""
    order, todo = [], [(root, None)]
    while todo:
        node, parent = todo.pop()
        order.append((node, parent))
        todo.extend((child, node) for child in reversed(node.children))
    return order


def _build_nwk_trees(roots: list[_NwkNode], key: str) -> dict[str, nx.DiGraph]:
    """Convert parsed Newick trees to a dictionary of named :class:`networkx.DiGraph`."""
    orders = [_preorder(root) for root in roots]
    # Internal labels that are numbers are support values, everything else is a name
    for order in orders:
        for node, _ in order:
            if not node.label:
                continue
            if node.children and not node.quoted and _NWK_NUMBER.fullmatch(node.label):
                node.support = float(node.label)
            else:
                node.name = node.label
    # Unnamed nodes get generated names that are unique across all trees
    taken = {node.name for order in orders for node, _ in order if node.name is not None}
    counter = itertools.count()
    graphs = {}
    for i, order in enumerate(orders):
        names = [node.name for node, _ in order if node.name is not None]
        duplicates = sorted(name for name, n in Counter(names).items() if n > 1)
        if duplicates:
            raise ValueError(f"Node names must be unique within a tree, but found duplicates: {duplicates[:10]}")
        # Branch lengths are only kept if the tree has any, including one on the root
        has_lengths = any(node.length is not None for node, _ in order)
        missing = False
        G = nx.DiGraph()
        for node, parent in order:
            while node.name is None:
                name = f"node{next(counter)}"
                if name not in taken:
                    node.name = name
                    taken.add(name)
            node_attrs = {} if node.support is None else {"support": node.support}
            edge_attrs = {}
            if has_lengths:
                if parent is None:
                    node.time = node.length if node.length is not None else 0.0
                else:
                    edge_attrs["length"] = node.length if node.length is not None else float("nan")
                    missing |= node.length is None
                    node.time = parent.time + edge_attrs["length"]
                node_attrs["time"] = node.time
            G.add_node(node.name, **node_attrs)
            if parent is not None:
                G.add_edge(parent.name, node.name, **edge_attrs)
        if missing:
            warnings.warn(
                "Some branches have no length. Their `length` is NaN and so is the `time` of their descendants.",
                UserWarning,
                stacklevel=3,
            )
        graphs[key if len(orders) == 1 else f"{key}_{i}"] = G
    return graphs


def read_nwk(
    filename: str | PathLike,
    *,
    key: str = "tree",
    alignment: Literal["leaves", "nodes"] = "leaves",
) -> TreeData:
    """Read a Newick (`.nwk`) tree file.

    The leaves of the tree (or all of its nodes, see `alignment`) become the observations of the
    returned :class:`~treedata.TreeData`, which has no variables, and the tree is stored in
    :attr:`~treedata.TreeData.obst`.

    Parameters
    ----------
    filename
        Path to a Newick file.
    key
        Key under which the tree is stored in :attr:`~treedata.TreeData.obst`. If the file
        contains several trees, they are stored as `{key}_0`, `{key}_1`, ... and the
        observations are the union of their nodes.
    alignment
        If `'leaves'`, only the leaves are observations. If `'nodes'`, all nodes are observations,
        including internal nodes and those with generated names. Sets
        :attr:`~treedata.TreeData.alignment` of the returned object.

    Returns
    -------
    A :class:`~treedata.TreeData` with the leaves or nodes as observations and no variables.

    Notes
    -----
    - If the tree has branch lengths, they are stored as the `length` attribute of the edges
      and the distance from the root as the `time` attribute of the nodes. The `time` of the
      root is its own branch length if it has one and 0 otherwise. If only some branches have
      a length (this includes a tree where only the root has one), the others get a `length`
      of NaN and a warning is raised. Without any branch lengths neither attribute is set.
    - Labels become node names. Internal labels that are numbers (e.g. bootstrap support) are
      stored as the `support` attribute of the node instead, unless they are quoted.
    - Nodes without a name get a unique generated name (`node0`, `node1`, ...).
    - Quoted labels (`'name with spaces'`) are supported. Underscores in unquoted labels are
      kept as they are.
    - Comments in square brackets, such as NHX or BEAST annotations, are ignored.
    - The semicolon after the last tree is optional.
    """
    if alignment not in ("leaves", "nodes"):
        raise ValueError("alignment has to be one of ['leaves', 'nodes']")
    with open(filename, encoding="utf-8-sig") as f:
        roots = _parse_newick(f.read())
    trees = _build_nwk_trees(roots, key)
    # A dict keeps the names unique and in the order of the file
    names = {
        node: None for tree in trees.values() for node in tree if alignment == "nodes" or tree.out_degree(node) == 0
    }
    return TreeData(obs=pd.DataFrame(index=pd.Index(list(names))), obst=trees, alignment=alignment)
