from __future__ import annotations

import json
import warnings
from collections.abc import MutableMapping
from importlib.metadata import version as get_version
from os import PathLike
from typing import Literal

import anndata as ad
import awkward as ak
import h5py
import networkx as nx
import zarr
from packaging import version

from treedata._core.treedata import TreeData

ANNDATA_VERSION = version.parse(get_version("anndata"))
USE_EXPERIMENTAL = ANNDATA_VERSION < version.parse("0.11.0")


def _read_elem(elem):
    """Read an element from a store."""
    if USE_EXPERIMENTAL:
        return ad.experimental.read_elem(elem)
    else:
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


def _decode_attr_column(arr) -> list:
    """Decode an attribute column to a Python list.

    Handles both awkward arrays (new format) and JSON-string numpy arrays (legacy format).
    """
    if isinstance(arr, ak.Array):
        return arr.to_list()
    return [json.loads(v) for v in arr]


def _read_axis_trees_zarr(g: zarr.Group) -> dict:
    """Read AxisTrees from a zarr group written in columnar format.

    None values are dropped when reconstructing node/edge attribute dicts,
    matching networkx semantics where an absent attribute differs from None.
    """
    trees = {}
    for name in g.keys():
        tg = g[name]
        if not isinstance(tg, zarr.Group):
            continue
        G = nx.DiGraph()
        nodes = list(_read_elem(tg["nodes"]))
        node_attrs: dict[str, list] = {}
        if "node_attrs" in tg:
            node_attrs = {k: _decode_attr_column(v) for k, v in _read_elem(tg["node_attrs"]).items()}
        for i, node in enumerate(nodes):
            G.add_node(node, **{k: v[i] for k, v in node_attrs.items() if v[i] is not None})
        edges = _read_elem(tg["edges"])
        edge_attrs: dict[str, list] = {}
        if "edge_attrs" in tg:
            edge_attrs = {k: _decode_attr_column(v) for k, v in _read_elem(tg["edge_attrs"]).items()}
        for i, (src, dst) in enumerate(edges):
            G.add_edge(src, dst, **{k: v[i] for k, v in edge_attrs.items() if v[i] is not None})
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
    for k in ["obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "label", "allow_overlap", "alignment"]:
        if k in f:
            d[k] = _read_elem(f[k])
    # Read raw
    if "raw" in f:
        d["raw"] = _read_raw(f, backed)
    # Read axis tree elements
    for k in ["obst", "vart"]:
        if k in f:
            elem = f[k]
            if isinstance(elem, zarr.Group):
                d[k] = _read_axis_trees_zarr(elem)
            else:
                d[k] = _parse_axis_trees(_read_elem(elem))
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
