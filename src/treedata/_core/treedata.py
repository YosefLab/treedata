from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from anndata._core.index import _subset
from scipy import sparse

from .aligned_mapping import (
    AxisTrees,
)

if TYPE_CHECKING:
    from os import PathLike

    Index1D = slice | int | str | np.int64 | np.ndarray
    Index = Index1D | tuple[Index1D, Index1D] | sparse.spmatrix | sparse.sparray


class TreeData(ad.AnnData):
    """AnnData with trees.

    :class:`~treedata.TreeData` is a light-weight wrapper around :class:`~anndata.AnnData`
    which adds two additional attributes, :attr:`obst` and :attr:`vart`, to
    store trees for observations and variables. A :class:`~treedata.TreeData`
    object can be used just like an :class:`~anndata.AnnData` object and stores a
    data matrix :attr:`X` together with annotations
    of observations :attr:`obs` (:attr:`obsm`, :attr:`obsp`, :attr:`obst`),
    variables :attr:`var` (:attr:`varm`, :attr:`varp`, :attr:`vart`),
    and unstructured annotations :attr:`uns`.

    Parameters
    ----------
    X
        A #observations × #variables data matrix. A view of the data is used if the
        data type matches, otherwise, a copy is made.
    obs
        Key-indexed one-dimensional observations annotation of length #observations.
    var
        Key-indexed one-dimensional variables annotation of length #variables.
    uns
        Key-indexed unstructured annotation.
    obsm
        Key-indexed multi-dimensional observations annotation of length #observations.
        If passing a :class:`~numpy.ndarray`, it needs to have a structured datatype.
    obst
        Key-indexed :class:`~networkx.DiGraph` trees leaf nodes in the observations axis.
    varm
        Key-indexed multi-dimensional variables annotation of length #variables.
        If passing a :class:`~numpy.ndarray`, it needs to have a structured datatype.
    vart
        Key-indexed :class:`~networkx.DiGraph` trees leaf nodes in the variables axis.
    layers
        Key-indexed multi-dimensional arrays aligned to dimensions of `X`.
    shape
        Shape tuple (#observations, #variables). Can only be provided if `X` is `None`.
    filename
        Name of backing file. See :class:`h5py.File`.
    filemode
        Open mode of backing file. See :class:`h5py.File`.
    asview
        Initialize as view. `X` has to be an TreeData object.
    label
        Columns in `.obs` and `.var` to place tree key in. Default is "tree".
        If it's None, no column is added.
    allow_overlap
        Whether overlapping trees are allowed. Default is False.
    """

    def __init__(
        self,
        X: np.ndarray | sparse.spmatrix | pd.DataFrame | None = None,
        obs: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        var: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        uns: Mapping[str, Any] | None = None,
        obsm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        obst: Mapping[str, nx.DiGraph] | None = None,
        varm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        vart: Mapping[str, nx.DiGraph] | None = None,
        layers: Mapping[str, np.ndarray | sparse.spmatrix] | None = None,
        raw: Mapping[str, Any] | None = None,
        dtype: np.dtype | type | str | None = None,
        shape: tuple[int, int] | None = None,
        filename: PathLike | None = None,
        filemode: Literal["r", "r+"] | None = None,
        asview: bool = False,
        label: str | None = "tree",
        allow_overlap: bool = False,
        *,
        obsp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        oidx: Index1D = None,
        vidx: Index1D = None,
    ):
        if asview:
            if not isinstance(X, TreeData):
                raise ValueError("If asview is True, X has to be an TreeData object")
            self._init_as_view(X, oidx, vidx)
        else:
            self._init_as_actual(
                X=X,
                obs=obs,
                var=var,
                uns=uns,
                obsm=obsm,
                varm=varm,
                obsp=obsp,
                varp=varp,
                obst=obst,
                vart=vart,
                raw=raw,
                layers=layers,
                dtype=dtype,
                shape=shape,
                filename=filename,
                filemode=filemode,
                label=label,
                allow_overlap=allow_overlap,
            )

    def _init_as_actual(
        self,
        X=None,
        obs=None,
        var=None,
        uns=None,
        obsm=None,
        varm=None,
        varp=None,
        obsp=None,
        obst=None,
        vart=None,
        raw=None,
        layers=None,
        dtype=None,
        shape=None,
        filename=None,
        filemode=None,
        label=None,
        allow_overlap=None,
    ):
        super()._init_as_actual(
            X=X,
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            varm=varm,
            varp=varp,
            obsp=obsp,
            raw=raw,
            layers=layers,
            dtype=dtype,
            shape=shape,
            filename=filename,
            filemode=filemode,
        )

        # init from TreeData
        if isinstance(X, TreeData):
            self._tree_label = X.label
            self._allow_overlap = X.allow_overlap
            self._obst = X.obst
            self._vart = X.vart

        # init from scratch
        else:
            if isinstance(label, str) or label is None:
                self._tree_label = label
            else:
                raise ValueError("label has to be a string or None")
            if isinstance(allow_overlap, bool) or isinstance(allow_overlap, np.bool_):
                self._allow_overlap = bool(allow_overlap)
            else:
                raise ValueError("allow_overlap has to be a boolean")
            self._obst = AxisTrees(self, 0, vals=obst)
            self._vart = AxisTrees(self, 1, vals=vart)

    def _init_as_view(self, tdata_ref: TreeData, oidx: Index, vidx: Index):
        super()._init_as_view(tdata_ref, oidx=oidx, vidx=vidx)

        # view of obst and vart
        self._obst = tdata_ref.obst._view(self, (oidx,))
        self._vart = tdata_ref.vart._view(self, (vidx,))

        # set attributes
        self._tree_label = tdata_ref._tree_label
        self._allow_overlap = tdata_ref._allow_overlap

    def obst_keys(self) -> list[str]:
        """List keys of variable annotation `obst`."""
        return list(self._obst.keys())

    def vart_keys(self) -> list[str]:
        """List keys of variable annotation `vart`."""
        return list(self._vart.keys())

    @property
    def obst(self) -> AxisTrees:
        """Tree annotation of observations

        Stores for each key a :class:`~networkx.DiGraph` with leaf nodes in
        :attr:`obs_names`. Is subset and pruned with `data` but behaves
        otherwise like a :term:`mapping`.
        """
        return self._obst

    @property
    def vart(self) -> AxisTrees:
        """Tree annotation of variables

        Stores for each key a :class:`~networkx.DiGraph` with leaf nodes in
        :attr:`var_names`. Is subset and pruned with `data` but behaves
        otherwise like a :term:`mapping`.
        """
        return self._vart

    @property
    def allow_overlap(self) -> bool:
        """Whether overlapping trees are allowed."""
        return self._allow_overlap

    @property
    def label(self) -> str | None:
        """Column in `.obs` and .`obs` with tree keys"""
        return self._tree_label

    @property
    def is_view(self) -> bool:
        """`True` if object is view of another TreeData object, `False` otherwise."""
        return self._is_view

    @obst.setter
    def obst(self, value):
        obst = AxisTrees(self, 0, vals=dict(value))
        self._obst = obst

    @vart.setter
    def vart(self, value):
        vart = AxisTrees(self, 1, vals=dict(value))
        self._vart = vart

    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"TreeData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in ["obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp", "obst", "vart"]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)

    def __getitem__(self, index: Index) -> TreeData:
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return TreeData(self, oidx=oidx, vidx=vidx, asview=True)

    def concatenate(self) -> None:
        """Concatenate deprecated, use `treedata.concat` instead."""
        raise NotImplementedError("Concatenation deprecated, use `treedata.concat` instead")

    def to_adata(self) -> ad.AnnData:
        """Convert this TreeData object to an AnnData object."""
        return ad.AnnData(self)

    def _mutated_copy(self, **kwargs):
        """Creating TreeData with attributes optionally specified via kwargs."""
        if self.isbacked:
            if "X" not in kwargs or (self.raw is not None and "raw" not in kwargs):
                raise NotImplementedError(
                    "This function does not currently handle backed objects "
                    "internally, this should be dealt with before."
                )
        new = {}
        new["label"] = self.label
        new["allow_overlap"] = self.allow_overlap

        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "obst", "vart", "layers"]:
            if key in kwargs:
                new[key] = kwargs[key]
            else:
                new[key] = getattr(self, key).copy()
        if "X" in kwargs:
            new["X"] = kwargs["X"]
        elif self._has_X():
            new["X"] = self.X.copy()
        if "uns" in kwargs:
            new["uns"] = kwargs["uns"]
        else:
            new["uns"] = deepcopy(self._uns)
        if "raw" in kwargs:
            new["raw"] = kwargs["raw"]
        elif self.raw is not None:
            new["raw"] = self.raw.copy()

        return TreeData(**new)

    def copy(self, filename: PathLike | None = None) -> TreeData:
        """Full copy, optionally on disk."""
        if not self.isbacked:
            if self.is_view and self._has_X():
                return self._mutated_copy(X=_subset(self._adata_ref.X, (self._oidx, self._vidx)).copy())
            else:
                return self._mutated_copy()
        else:
            from .read import read_h5ad

            if filename is None:
                raise ValueError(
                    "To copy an TreeData object in backed mode, "
                    "pass a filename: `.copy(filename='myfilename.h5ad')`. "
                    "To load the object into memory, use `.to_memory()`."
                )
            mode = self.file._filemode
            self.write_h5ad(filename)
            return read_h5ad(filename, backed=mode)

    def transpose(self) -> TreeData:
        """Transpose whole object

        Data matrix is transposed, observations and variables are interchanged.
        Ignores `.raw`.
        """
        adata = super().transpose()
        treedata_transpose = TreeData(
            adata,
            obst=self.vart.copy(),
            vart=self.obst.copy(),
            label=self.label,
            allow_overlap=self.allow_overlap,
        )
        return treedata_transpose

    T = property(transpose)

    def write_h5ad(
        self,
        filename: PathLike | None = None,
        compression: Literal["gzip", "lzf"] | None = None,
        compression_opts: int | Any = None,
        **kwargs,
    ):
        """Write `.h5ad`-formatted hdf5 file.

        Parameters
        ----------
        filename
            Filename of data file. Defaults to backing file.
        compression
            [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
        compression_opts
            [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
        """
        from .write import write_h5ad

        if filename is None and not self.isbacked:
            raise ValueError("Provide a filename!")
        if filename is None:
            filename = self.filename

        write_h5ad(Path(filename), self, compression=compression, compression_opts=compression_opts)

        if self.isbacked:
            self.file.filename = filename

    write = write_h5ad  # a shortcut and backwards compat

    def write_zarr(
        self,
        store: MutableMapping | PathLike,
        chunks: bool | int | tuple[int, ...] | None = None,
        **kwargs,
    ):
        """Write a hierarchical Zarr array store.

        Parameters
        ----------
        store
            The filename, a :class:`~typing.MutableMapping`, or a Zarr storage class.
        chunks
            Chunk shape.
        """
        from .write import write_zarr

        write_zarr(Path(store), self, chunks=chunks)

    def to_memory(self, copy=False) -> TreeData:
        """Return a new AnnData object with all backed arrays loaded into memory.

        Params
        ------
            copy:
                Whether the arrays that are already in-memory should be copied.
        """
        adata = super().to_memory(copy)
        tdata = TreeData(
            adata,
            obst=self.obst.copy(),
            vart=self.vart.copy(),
            label=self.label,
            allow_overlap=self.allow_overlap,
        )
        return tdata
