from __future__ import annotations

import json
import warnings
from collections.abc import MutableMapping
from importlib.metadata import version as get_version
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import awkward as ak
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import zarr
from packaging import version

from treedata._core.aligned_mapping import AxisTrees
from treedata._core.treedata import TreeData

ANNDATA_VERSION = version.parse(get_version("anndata"))
USE_EXPERIMENTAL = ANNDATA_VERSION < version.parse("0.11.0")
ZARR_V2 = version.parse(get_version("zarr")) < version.parse("3.0.0")


def _make_serializable(data: Any) -> Any:
    """Make a dictionary serializable."""
    if isinstance(data, dict):
        return {k: _make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list | tuple | set):
        return [_make_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic | np.number):
        return data.item()
    elif isinstance(data, pd.Series):
        return data.tolist()
    else:
        return data


def _write_elem(f, k, elem, *, dataset_kwargs) -> None:
    """Write an element to a storage group using anndata encoding."""
    if USE_EXPERIMENTAL:
        ad.experimental.write_elem(f, k, elem, dataset_kwargs=dataset_kwargs)
    else:
        ad.io.write_elem(f, k, elem, dataset_kwargs=dataset_kwargs)


def _digraph_to_dict(G: nx.DiGraph) -> dict:
    """Convert a networkx.DiGraph to a dictionary."""
    G = nx.DiGraph(G)
    edge_dict = nx.to_dict_of_dicts(G)
    # Get node data
    node_dict = {node: G.nodes[node] for node in G.nodes()}
    # Combine edge and node data in one dictionary
    graph_dict = {"edges": edge_dict, "nodes": node_dict}
    return graph_dict


def _serialize_axis_trees(trees: AxisTrees) -> str:
    """Serialize AxisTrees to a JSON string (used for HDF5)."""
    d = {k: _digraph_to_dict(v) for k, v in trees.items()}
    return json.dumps(_make_serializable(d))


def _write_axis_trees_zarr(f: zarr.Group, key: str, trees: AxisTrees) -> None:
    """Write AxisTrees to zarr using columnar awkward arrays (one array per attribute key).

    Stores each DiGraph as a subgroup with separate arrays for node IDs, edge
    pairs, and per-attribute columns. Each attribute column is an awkward array
    built with ak.from_iter, which preserves native types (float64, int64,
    var * string, etc.) without JSON encoding.
    """
    g = f.require_group(key)
    for name, G in trees.items():
        tg = g.require_group(str(name))
        nodes = [str(n) for n in G.nodes()]
        edges = [(str(u), str(v)) for u, v in G.edges()]
        _write_elem(tg, "nodes", np.array(nodes, dtype=object), dataset_kwargs={})
        edge_arr = np.array(edges, dtype=object) if edges else np.zeros((0, 2), dtype=object)
        _write_elem(tg, "edges", edge_arr, dataset_kwargs={})
        node_attr_keys = {k for n in G.nodes() for k in G.nodes[n]}
        if node_attr_keys:
            _write_elem(
                tg,
                "node_attrs",
                {k: ak.from_iter([_make_serializable(G.nodes[n].get(k)) for n in G.nodes()]) for k in node_attr_keys},
                dataset_kwargs={},
            )
        edge_attr_keys = {k for u, v in G.edges() for k in G[u][v]}
        if edge_attr_keys:
            _write_elem(
                tg,
                "edge_attrs",
                {k: ak.from_iter([_make_serializable(G[u][v].get(k)) for u, v in G.edges()]) for k in edge_attr_keys},
                dataset_kwargs={},
            )


def _write_tdata(f, tdata, filename, chunks=None, **kwargs) -> None:
    """Write TreeData to file."""
    # Add encoding type and version
    f = f["/"]
    f.attrs.setdefault("encoding-type", "treedata")
    f.attrs.setdefault("encoding-version", "0.1.0")
    # Convert strings to categoricals
    tdata.strings_to_categoricals()
    # Write X if not backed
    if not (tdata.isbacked and Path(tdata.filename) == Path(filename)):
        _write_elem(f, "X", tdata.X, dataset_kwargs=kwargs.update({"chunks": chunks}) if chunks else kwargs)
    # Write array elements
    for key in ["obs", "var", "label", "allow_overlap", "alignment"]:
        _write_elem(f, key, getattr(tdata, key), dataset_kwargs=kwargs)
    # Write group elements
    for key in ["obsm", "varm", "obsp", "varp", "layers", "uns"]:
        _write_elem(f, key, dict(getattr(tdata, key)), dataset_kwargs=kwargs)
    # Write axis tree elements
    for key in ["obst", "vart"]:
        if isinstance(f, zarr.Group):
            _write_axis_trees_zarr(f, key, getattr(tdata, key))
        else:
            _write_elem(f, key, _serialize_axis_trees(getattr(tdata, key)), dataset_kwargs=kwargs)
    # Write raw
    if tdata.raw is not None:
        tdata.strings_to_categoricals(tdata.raw.var)
        _write_elem(f, "raw", tdata.raw, dataset_kwargs=kwargs)
    # Close the file
    tdata.file.close()


def write_h5td(
    filename: PathLike | None,
    tdata: TreeData,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | Any = None,
    **kwargs,
) -> None:
    """Write `.h5td`-formatted hdf5 file.

    Parameters
    ----------
    filename
        Filename of data file. Defaults to backing file.
    tdata
        TreeData object to write.
    compression
        [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
    compression_opts
        [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
    """
    mode = "a" if tdata.isbacked else "w"
    if tdata.isbacked:  # close so that we can reopen below
        tdata.file.close()
    with h5py.File(filename, mode) as f:
        _write_tdata(f, tdata, filename, compression=compression, compression_opts=compression_opts, **kwargs)


def write_h5ad(
    filename: PathLike | None,
    tdata: TreeData,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | Any = None,
    **kwargs,
) -> None:
    """Write `.h5td`-formatted hdf5 file. Deprecated, use `write_h5td` instead.

    Parameters
    ----------
    filename
        Filename of data file. Defaults to backing file.
    tdata
        TreeData object to write.
    compression
        [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
    compression_opts
        [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
    """
    warnings.warn(
        "write_h5ad has been renamed to write_h5td. write_h5ad will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    write_h5td(
        filename=filename,
        tdata=tdata,
        compression=compression,
        compression_opts=compression_opts,
        **kwargs,
    )


def _open_zarr_group(
    store: MutableMapping | PathLike | zarr.Group,
    *,
    mode: str = "w",
) -> tuple[zarr.Group, bool]:
    """Open a Zarr group for writing and indicate ownership."""
    if isinstance(store, zarr.Group):
        return store, False
    group_kwargs: dict[str, int] = {}
    if not ZARR_V2:
        group_kwargs["zarr_format"] = 3
    return zarr.open_group(store, mode=mode, **group_kwargs), True


def _close_zarr_group(group: zarr.Group) -> None:
    """Close the underlying store for an opened Zarr group if needed."""
    store = getattr(group, "store", None)
    if store is not None:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _is_zip_store(store: Any) -> bool:
    """Return True if store is or resolves to a zarr ZipStore."""
    if ZARR_V2:
        return False
    return isinstance(store, zarr.storage.ZipStore) or (
        isinstance(store, (str, PathLike)) and str(store).endswith(".zip")
    )


def write_zarr(
    filename: MutableMapping | PathLike | zarr.Group, tdata: TreeData, chunks: tuple[int, ...] | None = None, **kwargs
) -> None:
    """Write `.zarr`-formatted zarr file.

    Parameters
    ----------
    filename
        Filename of data file. Defaults to backing file.
    tdata
        TreeData object to write.
    kwargs
        Additional keyword arguments passed to :func:`zarr.save`.
    """
    if _is_zip_store(filename):
        _write_zarr_via_memory(filename, tdata, chunks, **kwargs)
    else:
        f, should_close = _open_zarr_group(filename)
        _write_tdata(f, tdata, filename, chunks, **kwargs)
        if should_close:
            _close_zarr_group(f)


def _write_zarr_via_memory(
    zip_target: zarr.storage.ZipStore | PathLike | str,
    tdata: TreeData,
    chunks: tuple[int, ...] | None,
    **kwargs: Any,
) -> None:
    """Write to a ZipStore via an intermediate MemoryStore.

    zarr v3's ZipStore is append-only: every attribute update rewrites zarr.json,
    producing duplicate-name warnings. Writing to MemoryStore first (which
    supports overwriting) and then bulk-copying the final state to the ZipStore
    ensures each key is written exactly once.
    """
    mem_store = zarr.storage.MemoryStore()
    f = zarr.open_group(mem_store, mode="w", zarr_format=3)
    _write_tdata(f, tdata, zip_target, chunks, **kwargs)

    if isinstance(zip_target, zarr.storage.ZipStore):
        zip_store = zip_target
        should_close = False
    else:
        zip_store = zarr.storage.ZipStore(str(zip_target), mode="w")
        should_close = True

    # Copy every key from MemoryStore to ZipStore exactly once.
    # ZipStore has no __setitem__; writes go through the async set() method.
    # zarr.core.sync.sync is the same utility zarr uses for its own sync API.
    from zarr.core.sync import sync as _zarr_sync

    async def _bulk_copy() -> None:
        for key, value in mem_store._store_dict.items():
            await zip_store.set(key, value)

    _zarr_sync(_bulk_copy())

    if should_close:
        zip_store.close()
