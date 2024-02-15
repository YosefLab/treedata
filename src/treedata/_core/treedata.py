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
from anndata._core.index import Index
from scipy import sparse

from .aligned_mapping import (
    AxisTrees,
)

if TYPE_CHECKING:
    from os import PathLike


class TreeData(ad.AnnData):
    """AnnData with trees.

    `TreeData` is a light-weight wrapper around :class:`~anndata.AnnData`
    which adds two additional attributes, `obst` and `vart`, to
    store trees for observations and variables A `TreeData`
    object can be used just like an :class:`~anndata.AnnData` object and stores a
    data matrix `X` together with annotations
    of observations `obs` (`obsm`, `obsp`, `obst`),
    variables `var` (`varm`, `varp`, `vart`),
    and unstructured annotations `uns`.

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
        Column in `.obs` to place tree id in. Default is "tree".
        If it's None, no column is added.
    allow_tree_overlap
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
        allow_tree_overlap: bool = False,
    ):
        super().__init__(
            X=X,
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            varm=varm,
            raw=raw,
            layers=layers,
            dtype=dtype,
            shape=shape,
            filename=filename,
            filemode=filemode,
            asview=asview,
        )

        if label is not None:
            if label in self.obs.columns:
                warnings.warn(f"Tree label {label} already present in .obs overwriting it", stacklevel=2)
            self.obs[label] = pd.NA

        self._tree_label = label
        self._allow_tree_overlap = allow_tree_overlap

        self._obst = AxisTrees(self, 0, vals=obst)
        self._vart = AxisTrees(self, 1, vals=vart)

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
        `obs_names`. Is subset and pruned with `data` but behaves
        otherwise like a :term:`mapping`.
        """
        return self._obst

    @property
    def vart(self):
        """Tree annotation of variables

        Stores for each key a :class:`~networkx.DiGraph` with leaf nodes in
        `var_names`. Is subset and pruned with `data` but behaves
        otherwise like a :term:`mapping`.
        """
        return self._vart

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
        raise NotImplementedError("Slicing not yet implemented")
