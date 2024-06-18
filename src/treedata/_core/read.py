from __future__ import annotations

import json
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import (
    Literal,
)

import anndata as ad
import h5py
import zarr
from scipy import sparse

from treedata._core.aligned_mapping import AxisTrees
from treedata._core.treedata import TreeData
from treedata._utils import dict_to_digraph


def _tdata_from_adata(tdata, treedata_attrs=None) -> TreeData:
    """Create a TreeData object parsing attribute from AnnData uns field."""
    tdata.__class__ = TreeData
    if treedata_attrs is not None:
        tdata._tree_label = treedata_attrs["label"] if "label" in treedata_attrs.keys() else None
        tdata._allow_overlap = bool(treedata_attrs["allow_overlap"])
        tdata._obst = AxisTrees(tdata, 0, vals={k: dict_to_digraph(v) for k, v in treedata_attrs["obst"].items()})
        tdata._vart = AxisTrees(tdata, 1, vals={k: dict_to_digraph(v) for k, v in treedata_attrs["vart"].items()})
    else:
        tdata._tree_label = None
        tdata._allow_overlap = False
        tdata._obst = AxisTrees(tdata, 0)
        tdata._vart = AxisTrees(tdata, 1)
    return tdata


def read_h5ad(
    filename: str | Path = None,
    backed: Literal["r", "r+"] | bool | None = None,
    *,
    as_sparse: Sequence[str] = (),
    as_sparse_fmt: type[sparse.spmatrix] = sparse.csr_matrix,
    chunk_size: int = 6000,
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
    as_sparse
        If an array was saved as dense, passing its name here will read it as
        a sparse_matrix, by chunk of size `chunk_size`.
    as_sparse_fmt
        Sparse format class to read elements from `as_sparse` in as.
    chunk_size
        Used only when loading sparse dataset that is stored as dense.
        Loading iterates through chunks of the dataset of this row size
        until it reads the whole dataset.
        Higher size means higher memory consumption and higher (to a point)
        loading speed.
    """
    adata = ad.read_h5ad(
        filename,
        backed=backed,
        as_sparse=as_sparse,
        as_sparse_fmt=as_sparse_fmt,
        chunk_size=chunk_size,
    )
    with h5py.File(filename, "r") as f:
        if "raw.treedata" in f:
            treedata_attrs = json.loads(f["raw.treedata"][()])
        else:
            treedata_attrs = None
    tdata = _tdata_from_adata(adata, treedata_attrs)

    return tdata


def read_zarr(store: str | Path | MutableMapping | zarr.Group) -> TreeData:
    """Read from a hierarchical Zarr array store.

    Parameters
    ----------
    store
        The filename, a :class:`~typing.MutableMapping`, or a Zarr storage class.
    """
    adata = ad.read_zarr(store)

    with zarr.open(store, mode="r") as f:
        if "raw.treedata" in f:
            treedata_attrs = json.loads(f["raw.treedata"][()])
        else:
            treedata_attrs = None
    tdata = _tdata_from_adata(adata, treedata_attrs)

    return tdata
