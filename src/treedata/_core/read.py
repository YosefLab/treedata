from __future__ import annotations

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import (
    Literal,
)

import anndata as ad
import h5py
import networkx as nx
import zarr
from packaging import version

from treedata._core.treedata import TreeData

ANDATA_VERSION = version.parse(ad.__version__)
USE_EXPERIMENTAL = ANDATA_VERSION < version.parse("0.11.0")


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
    """Parse AxisTrees from a string."""
    return {k: _dict_to_digraph(v) for k, v in json.loads(data).items()}


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
    for k in ["obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "label", "allow_overlap"]:
        if k in f:
            d[k] = _read_elem(f[k])
    # Read raw
    if "raw" in f:
        d["raw"] = _read_raw(f, backed)
    # Read axis tree elements
    for k in ["obst", "vart"]:
        if k in f:
            d[k] = _parse_axis_trees(_read_elem(f[k]))
    # Read legacy treedata format
    if "raw.treedata" in f:
        d.update(_parse_legacy(json.loads(_read_elem(f["raw.treedata"]))))
    return d


def read_h5ad(
    filename: str | Path = None,
    backed: Literal["r", "r+"] | bool | None = None,
) -> TreeData:
    """Read `.h5ad`-formatted hdf5 file.

    Parameters
    ----------
    filename
        File name of data file.
    backed
        If `'r'`, load :class:`~anndata.TreeData` in `backed` mode
        instead of fully loading it into memory (`memory` mode).
        If you want to modify backed attributes of the TreeData object,
        you need to choose `'r+'`.
    """
    with h5py.File(filename, "r") as f:
        d = _read_tdata(f, filename, backed)
    return TreeData(**d)


def read_zarr(store: str | Path | MutableMapping | zarr.Group) -> TreeData:
    """Read from a hierarchical Zarr array store.

    Parameters
    ----------
    store
        The filename, a :class:`~typing.MutableMapping`, or a Zarr storage class.
    """
    with zarr.open(store, mode="r") as f:
        d = _read_tdata(f, store, backed=False)
    return TreeData(**d)
