from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from anndata._core.index import _subset
from scipy import sparse

from treedata._utils import _get_nodes

from .aligned_mapping import AxisTrees, AxisTreesView

if TYPE_CHECKING:
    from os import PathLike

    Index1D = slice | int | str | np.int64 | np.ndarray | list[str | int] | pd.Index
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
    alignment
        Alignment between trees and observations/variables. One of the following:

        - `leaves`: All leaf names are present in the observation/variable names.
        - `nodes`: All leaf and internal node names are present in the observation/variable names.
        - `subset`: A subset of leaf and internal node names are present in the observation/variable names.
    allow_overlap
        Whether trees containing overlapping sets of leaves or nodes are allowed. Default is True.
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
        alignment: Literal["leaves", "nodes", "subset"] = "leaves",
        allow_overlap: bool = True,
        *,
        obsp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        oidx: Index1D | None = None,
        vidx: Index1D | None = None,
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
                alignment=alignment,
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
        alignment=None,
        allow_overlap=None,
    ):
        # init tree only
        if X is None and obs is None and obst is not None:
            obs = pd.DataFrame(index=_get_nodes(obst, alignment))
        if X is None and var is None and vart is not None:
            var = pd.DataFrame(index=_get_nodes(vart, alignment))

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
            self._alignment = X._alignment
            self._has_overlap = X.has_overlap

        # init from scratch
        else:
            if isinstance(label, str) or label is None:
                self._tree_label = label
            else:
                raise ValueError("label has to be a string or None")
            if alignment not in ["leaves", "nodes", "subset"]:
                raise ValueError("alignment has to be one of ['leaves', 'nodes', 'subset']")
            else:
                self._alignment = alignment
            if isinstance(allow_overlap, bool) or isinstance(allow_overlap, np.bool_):
                self._allow_overlap = bool(allow_overlap)
            else:
                raise ValueError("allow_overlap has to be a boolean")
            self._has_overlap = False
            self._obst = AxisTrees(self, 0, vals=obst)
            self._vart = AxisTrees(self, 1, vals=vart)
            self._update_has_overlap()

    def _init_as_view(self, tdata_ref: TreeData, oidx: Index1D | None, vidx: Index1D | None):
        super()._init_as_view(tdata_ref, oidx=oidx, vidx=vidx)

        # set attributes
        self._tree_label = tdata_ref._tree_label
        self._alignment = tdata_ref._alignment
        self._allow_overlap = tdata_ref._allow_overlap
        self._has_overlap = tdata_ref._has_overlap

        # view of obst and vart
        self._obst = tdata_ref.obst._view(self, oidx)
        self._vart = tdata_ref.vart._view(self, vidx)

        # actualize if not a valid subset
        for attr in ["obst", "vart"]:
            if not getattr(self, attr)._validate_mapping():
                warnings.warn(
                    f"One or more trees in `{attr}` are not disconnected by subsetting. Changing alignment to `subset` and initializing view as actual.",
                    stacklevel=2,
                )
                self._alignment = "subset"
                new = self.copy()
                self._init_as_actual(new)
                break

    def obst_keys(self) -> list[str]:
        """List keys of variable annotation `obst`."""
        return list(self._obst.keys())

    def vart_keys(self) -> list[str]:
        """List keys of variable annotation `vart`."""
        return list(self._vart.keys())

    @property
    def obst(self) -> AxisTrees | AxisTreesView:
        """Tree annotation of observations

        Stores for each key a :class:`~networkx.DiGraph` with leaf nodes in
        :attr:`obs_names`. Is subset and pruned with `data` but behaves
        otherwise like a :term:`alignment`.
        """
        return self._obst

    @property
    def vart(self) -> AxisTrees | AxisTreesView:
        """Tree annotation of variables

        Stores for each key a :class:`~networkx.DiGraph` with leaf nodes in
        :attr:`var_names`. Is subset and pruned with `data` but behaves
        otherwise like a :term:`alignment`.
        """
        return self._vart

    @property
    def allow_overlap(self) -> bool:
        """Whether overlapping trees are allowed."""
        return self._allow_overlap

    @property
    def has_overlap(self) -> bool:
        """
        Flag indicating whether stored trees contain overlapping nodes.

        Returns
        -------
        bool - ``True`` when any stored trees share nodes, ``False`` otherwise.
        """
        return self._has_overlap

    @property
    def alignment(self) -> Literal["leaves", "nodes", "subset"]:
        """Mapping between trees and observations/variables."""
        return self._alignment  # type: ignore

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
        self._update_has_overlap()

    @vart.setter
    def vart(self, value):
        vart = AxisTrees(self, 1, vals=dict(value))
        self._vart = vart
        self._update_has_overlap()

    @label.setter
    def label(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("label has to be a string or None")
        self._tree_label = value

    @allow_overlap.setter
    def allow_overlap(self, value):
        if not isinstance(value, bool):
            raise ValueError("allow_overlap has to be a boolean")
        if not value:
            for attr in ["obst", "vart"]:
                if getattr(self, attr)._check_tree_overlap():
                    raise ValueError(
                        f"One or more trees in {attr} have overlapping nodes. Cannot set allow_overlap to False."
                    )
        self._allow_overlap = value
        self._update_has_overlap()

    def _update_has_overlap(self) -> None:
        """
        Update the cached overlap indicator.

        Ensures the cached `_has_overlap` flag matches the current state of stored
        trees.

        Parameters
        ----------
        None
            This method does not accept any parameters.

        Returns
        -------
        None - This function updates the `_has_overlap` attribute in place.
        """
        if not self._allow_overlap:
            self._has_overlap = False
            return

        has_overlap = False
        if hasattr(self, "_obst"):
            has_overlap = has_overlap or self._obst._check_tree_overlap()
        if hasattr(self, "_vart"):
            has_overlap = has_overlap or self._vart._check_tree_overlap()
        self._has_overlap = has_overlap

    @alignment.setter
    def alignment(self, value):
        if value not in ["leaves", "nodes", "subset"]:
            raise ValueError("alignment has to be one of ['leaves', 'nodes', 'subset']")
        previous = self._alignment
        self._alignment = value
        for attr in ["obst", "vart"]:
            if not getattr(self, attr)._validate_mapping():
                self._alignment = previous
                raise ValueError(f"One or more trees in `{attr}` cannot be transitioned to {value} alignment.")

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

    def __getitem__(self, index: Any) -> TreeData:
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
        new["alignment"] = self.alignment

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
            from .read import read_h5td

            if filename is None:
                raise ValueError(
                    "To copy an TreeData object in backed mode, "
                    "pass a filename: `.copy(filename='myfilename.h5td')`. "
                    "To load the object into memory, use `.to_memory()`."
                )
            mode = self.file._filemode
            self.write_h5td(filename)
            return read_h5td(filename, backed=mode)

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
            alignment=self.alignment,
        )
        return treedata_transpose

    T = property(transpose)

    def write_h5td(
        self,
        filename: PathLike | None = None,
        compression: Literal["gzip", "lzf"] | None = None,
        compression_opts: int | Any = None,
        **kwargs,
    ):
        """Write `.h5td`-formatted hdf5 file.

        Parameters
        ----------
        filename
            Filename of data file. Defaults to backing file.
        compression
            [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
        compression_opts
            [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
        """
        from .write import write_h5td

        if filename is None and not self.isbacked:
            raise ValueError("Provide a filename!")
        if filename is None:
            filename = self.filename

        write_h5td(filename, self, compression=compression, compression_opts=compression_opts)

        if self.isbacked:
            self.file.filename = filename

    write = write_h5td  # a shortcut and backwards compat

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

        write_zarr(store, self, chunks=chunks)

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
            alignment=self.alignment,
        )
        return tdata
