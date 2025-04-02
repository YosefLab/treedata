from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import anndata as ad
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import zarr
from packaging import version

from treedata._core.aligned_mapping import AxisTrees
from treedata._core.treedata import TreeData

ANDATA_VERSION = version.parse(ad.__version__)
USE_EXPERIMENTAL = ANDATA_VERSION < version.parse("0.11.0")


def _make_serializable(data: dict) -> dict:
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


def _serialize_axis_trees(trees: AxisTrees) -> dict:
    """Serialize AxisTrees."""
    d = {k: _digraph_to_dict(v) for k, v in trees.items()}
    return json.dumps(_make_serializable(d))


def _write_tdata(f, tdata, filename, **kwargs) -> None:
    """Write TreeData to file."""
    # Add encoding type and version
    f = f["/"]
    f.attrs.setdefault("encoding-type", "anndata")
    f.attrs.setdefault("encoding-version", "0.1.0")
    # Convert strings to categoricals
    tdata.strings_to_categoricals()
    # Write X if not backed
    if not (tdata.isbacked and Path(tdata.filename) == Path(filename)):
        _write_elem(f, "X", tdata.X, dataset_kwargs=kwargs)
    # Write array elements
    for key in ["obs", "var", "label", "allow_overlap"]:
        _write_elem(f, key, getattr(tdata, key), dataset_kwargs=kwargs)
    # Write group elements
    for key in ["obsm", "varm", "obsp", "varp", "layers", "uns"]:
        _write_elem(f, key, dict(getattr(tdata, key)), dataset_kwargs=kwargs)
    # Write axis tree elements
    for key in ["obst", "vart"]:
        _write_elem(f, key, _serialize_axis_trees(getattr(tdata, key)), dataset_kwargs=kwargs)
    # Write raw
    if tdata.raw is not None:
        tdata.strings_to_categoricals(tdata.raw.var)
        _write_elem(f, "raw", tdata.raw, dataset_kwargs=kwargs)
    # Close the file
    tdata.file.close()


def write_h5ad(
    filename: str | Path,
    tdata: TreeData,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | Any = None,
    **kwargs,
) -> None:
    """Write `.h5ad`-formatted hdf5 file.

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


def write_zarr(filename: str | Path, tdata: TreeData, **kwargs) -> None:
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
    with zarr.open(filename, mode="w") as f:
        _write_tdata(f, tdata, filename, **kwargs)
