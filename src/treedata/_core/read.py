from __future__ import annotations

import json
import warnings
from collections.abc import MutableMapping
from os import PathLike
from typing import Literal

import anndata as ad
import h5py
import networkx as nx
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
