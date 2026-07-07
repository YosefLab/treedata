from __future__ import annotations

import json
import warnings
from collections.abc import MutableMapping
from importlib.metadata import version as get_version
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import anndata as ad
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

# Sentinel distinguishing an absent attribute from an attribute explicitly set to ``None``.
_MISSING = object()


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


def _encode_attr_column(values: list) -> np.ndarray:
    """Encode one attribute column (values in node/edge order) as a 1-D array.

    A column is stored with a native numpy dtype when it is *dense* (every
    node/edge has the attribute) and every value is a scalar of a single
    numeric or boolean Python type. This keeps the common case (e.g. ``depth``,
    ``time``, branch ``length``) compact and typed.

    Otherwise — ragged values (``list``/``set``/``np.ndarray``/nested), mixed
    types, string values, sparse columns, or explicit ``None`` — the column
    falls back to an object array of per-element JSON strings. The empty string
    ``""`` is reserved as the "attribute absent" marker, so an attribute
    explicitly set to ``None`` (stored as JSON ``"null"``) is preserved and kept
    distinct from a missing attribute.

    The number of attribute keys is expected to be small, so this stores only a
    handful of arrays per tree — object count scales with the number of keys,
    never with the number of nodes/edges.
    """
    dense = all(v is not _MISSING for v in values)
    serialized = [v if v is _MISSING else _make_serializable(v) for v in values]
    present_types = {type(v) for v in serialized if v is not _MISSING}
    # Native path: dense column of a single numeric/bool Python type.
    if dense and len(present_types) == 1 and next(iter(present_types)) in (int, float, bool):
        return np.array(serialized)
    return np.array(["" if v is _MISSING else json.dumps(v) for v in serialized], dtype=object)


def _write_axis_trees_zarr(f: zarr.Group, key: str, trees: AxisTrees) -> None:
    """Write AxisTrees to zarr in a columnar layout (one array per attribute key).

    Each ``DiGraph`` is stored as a subgroup with separate arrays for node IDs,
    edge pairs, and per-attribute columns (see :func:`_encode_attr_column`).
    This avoids the previous single-JSON-scalar format, which zarr v3 could not
    store above a size limit and which crashed on large trees.
    """
    # Remove any previously written trees so the persisted set matches ``trees``
    # exactly (e.g. when rewriting into an existing group).
    if key in f:
        del f[key]
    g = f.require_group(key)
    for name, G in trees.items():
        tg = g.require_group(str(name))
        nodes = list(G.nodes())
        edges = list(G.edges())
        node_ids = np.array([str(n) for n in nodes], dtype=object)
        edge_ids = np.array([(str(u), str(v)) for u, v in edges], dtype=object) if edges else np.zeros((0, 2), object)
        _write_elem(tg, "nodes", node_ids, dataset_kwargs={})
        _write_elem(tg, "edges", edge_ids, dataset_kwargs={})
        node_attr_keys = {k for n in nodes for k in G.nodes[n]}
        if node_attr_keys:
            cols = {k: _encode_attr_column([G.nodes[n].get(k, _MISSING) for n in nodes]) for k in node_attr_keys}
            _write_elem(tg, "node_attrs", cols, dataset_kwargs={})
        edge_attr_keys = {k for u, v in edges for k in G[u][v]}
        if edge_attr_keys:
            cols = {k: _encode_attr_column([G[u][v].get(k, _MISSING) for u, v in edges]) for k in edge_attr_keys}
            _write_elem(tg, "edge_attrs", cols, dataset_kwargs={})


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
    for key in ["obsm", "varm", "obsp", "varp", "uns"]:
        _write_elem(f, key, dict(getattr(tdata, key)), dataset_kwargs=kwargs)
    # Write layers without X (X is written separately above; None key added in anndata 0.13.x)
    _write_elem(f, "layers", {k: v for k, v in tdata.layers.items() if k is not None}, dataset_kwargs=kwargs)
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


def _write_zarr_via_memory(
    zip_target: zarr.storage.ZipStore | PathLike | str,
    tdata: TreeData,
    chunks: tuple[int, ...] | None,
    **kwargs: Any,
) -> None:
    """Write to a ZipStore via an intermediate MemoryStore.

    zarr v3's ZipStore is append-only: every metadata update appends a new
    zarr.json entry, producing duplicate-name warnings. Writing to MemoryStore
    first (which supports overwriting) and then bulk-copying the final state to
    the ZipStore ensures each key is written exactly once.
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

    # ZipStore has no __setitem__; writes go through the async set() method.
    from zarr.core.sync import sync as _zarr_sync

    async def _bulk_copy() -> None:
        for key, value in mem_store._store_dict.items():
            await zip_store.set(key, value)

    _zarr_sync(_bulk_copy())

    if should_close:
        zip_store.close()


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
