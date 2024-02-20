from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from anndata._core.index import Index, Index1D
from scipy import sparse

from .aligned_mapping import (
    AxisTrees,
)

if TYPE_CHECKING:
    from os import PathLike


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
        Initialize as view. `X` has to be an AnnData object.
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
            if label is not None:
                for attr in ["obs", "var"]:
                    if label in getattr(self, attr).columns:
                        warnings.warn(f"label {label} already present in .{attr} overwriting it", stacklevel=2)
                        getattr(self, attr)[label] = pd.NA
            self._tree_label = label
            self._allow_overlap = allow_overlap
            self._obst = AxisTrees(self, 0, vals=obst)
            self._vart = AxisTrees(self, 1, vals=vart)

    def _init_as_view(self, tdata_ref: TreeData, oidx: Index, vidx: Index):
        super()._init_as_view(tdata_ref, oidx, vidx)

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
    def vart(self):
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

    @obst.setter
    def obst(self, value):
        obst = AxisTrees(self, 0, vals=dict(value))
        self._obst = obst

    @vart.setter
    def vart(self, value):
        vart = AxisTrees(self, 0, vals=dict(value))
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

    def copy(self) -> TreeData:
        """Full copy of the object."""
        adata = super().copy()

        # remove label from obs and var
        if self.label is not None:
            if self.label in adata.obs.columns:
                adata.obs.drop(columns=self.label, inplace=True)
            if self.label in adata.var.columns:
                adata.var.drop(columns=self.label, inplace=True)
        # create a new TreeData object
        treedata_copy = TreeData(
            adata,
            obst=self.obst.copy(),
            vart=self.vart.copy(),
            label=self.label,
            allow_overlap=self.allow_overlap,
        )
        return treedata_copy
