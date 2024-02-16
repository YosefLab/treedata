from __future__ import annotations

import warnings
from collections import abc as cabc
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
    Union,
)

import anndata as ad
import networkx as nx
import pandas as pd

from treedata._utils import get_leaves, get_root, subset_tree

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.raw import Raw

    from .treedata import TreeData

OneDIdx = Union[Sequence[int], Sequence[bool], slice]
TwoDIdx = tuple[OneDIdx, OneDIdx]

I = TypeVar("I", OneDIdx, TwoDIdx, covariant=True)


class AxisTreesBase(cabc.MutableMapping):
    """Mapping of key to nx.DiGraph aligned to an axis of parent TreeData."""

    def __repr__(self):
        return f"{type(self).__name__} with keys: {', '.join(self.keys())}"

    def _ipython_key_completions_(self) -> list[str]:
        return list(self.keys())

    def _validate_value(self, value: nx.DiGraph, key: str) -> nx.DiGraph:
        # Check value type
        if not isinstance(value, nx.DiGraph):
            raise ValueError(f"Tree for key {key} must be a nx.DiGraph")
        # Check acyclic
        if not nx.is_directed_acyclic_graph(value):
            raise ValueError(f"Tree for key {key} cannot have cycles")
        # Check fully connected
        if not nx.is_weakly_connected(value):
            raise ValueError(f"Tree for key {key} must be fully connected")
        # Check alignment
        leaves = get_leaves(value)
        if not set(leaves).issubset(self.dim_names):
            raise ValueError(f"Leaf nodes of tree for key {key} must be in {self.dim}_names")
        # Check root
        _ = get_root(value)
        # Check overlap
        if not self.parent.allow_overlap:
            if set(leaves).intersection(self._membership.keys()):
                raise ValueError(f"Leaf nodes of tree for key {key} overlap with other trees")
        return value

    def _update_tree_labels(self):
        if self.parent._tree_label is not None:
            if self.parent.allow_overlap:
                mapping = self._membership
            else:
                mapping = {k: v[0] for k, v in self._membership.items()}
            getattr(self.parent, self.dim)[self.parent._tree_label] = getattr(self.parent, f"{self.dim}_names").map(
                mapping
            )

    def copy(self):
        d = AxisTrees(self.parent, self._axis)
        for k, v in self.items():
            d[k] = v.copy()
        return d

    def _view(self, parent: TreeData, subset_idx: I):
        """Returns a subset copy-on-write view of the object."""
        return AxisTreesView(self, parent, subset_idx)

    @property
    def parent(self) -> AnnData | Raw:
        return self._parent

    @property
    def attrname(self) -> str:
        return f"{self.dim}m"

    @property
    def axes(self) -> tuple[Literal[0, 1]]:
        """Axes of the parent this is aligned to"""
        return (self._axis,)

    @property
    def dim(self) -> str:
        """Name of the dimension this aligned to."""
        return self._dimnames[self._axis]

    @property
    def dim_names(self) -> pd.Index:
        return (self.parent.obs_names, self.parent.var_names)[self._axis]


class AxisTrees(AxisTreesBase):
    def __init__(
        self,
        parent: ad.AnnData,
        axis: int,
        vals: Mapping | None = None,
    ):
        self._parent = parent
        self._dimnames = ("obs", "var")
        if axis not in (0, 1):
            raise ValueError()
        self._axis = axis
        self._data = {}
        self._membership = defaultdict(list)
        if vals is not None:
            self.update(vals)

    def __getitem__(self, key: str) -> nx.DiGraph:
        return self._data[key]

    def __setitem__(self, key: str, value: nx.DiGraph):
        value = self._validate_value(value, key)

        leaves = get_leaves(value)
        for leaf in leaves:
            self._membership[leaf].append(key)

        if not self.parent.is_view:
            self._update_tree_labels()
        self._data[key] = value

    def __delitem__(self, key: str):
        leaves = get_leaves(self._data[key])
        for leaf in leaves:
            self._membership[leaf].remove(key)
            if not self._membership[leaf]:
                del self._membership[leaf]

        self._update_tree_labels()
        del self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)


class AxisTreesView(AxisTreesBase):
    def __init__(
        self,
        parent_mapping: AxisTreesBase,
        parent_view: TreeData,
        subset_idx: OneDIdx,
    ):
        self.parent_mapping = parent_mapping
        self._parent = parent_view
        self._dimnames = ("obs", "var")
        self.subset_idx = subset_idx
        self._axis = parent_mapping._axis

    def __getitem__(self, key: str) -> nx.DiGraph:
        # Consider caching the subset trees
        leaves = get_leaves(self.parent_mapping[key])
        subset_leaves = set(leaves).intersection(self.dim_names.values)
        return subset_tree(self.parent_mapping[key], subset_leaves, asview=True)

    def __setitem__(self, key: str, value: nx.DiGraph):
        value = self._validate_value(value, key)  # Validate before mutating
        warnings.warn(
            f"Setting element `.{self.attrname}['{key}']` of view, " "initializing view as actual.", stacklevel=2
        )
        with view_update(self.parent, self.attrname, ()) as new_mapping:
            new_mapping[key] = value

    def __delitem__(self, key: str):
        if key not in self:
            raise KeyError(
                "'{key!r}' not found in view of {self.attrname}"
            )  # Make sure it exists before bothering with a copy
        warnings.warn(
            f"Removing element `.{self.attrname}['{key}']` of view, " "initializing view as actual.", stacklevel=2
        )
        with view_update(self.parent, self.attrname, ()) as new_mapping:
            del new_mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.parent_mapping)

    def __contains__(self, key: str) -> bool:
        return key in self.parent_mapping

    def __len__(self) -> int:
        return len(self.parent_mapping)


def view_update(tdata_view: TreeData, attr_name: str, keys: tuple[str, ...]):
    """Context manager for updating a view of an AnnData object.

    Contains logic for "actualizing" a view. Yields the object to be modified in-place.

    Parameters
    ----------
    adata_view
        A view of an AnnData
    attr_name
        Name of the attribute being updated
    keys
        Keys to the attribute being updated

    Yields
    ------
    `adata.attr[key1][key2][keyn]...`
    """
    new = TreeData.copy()
    attr = getattr(new, attr_name)
    container = reduce(lambda d, k: d[k], keys, attr)
    yield container
    tdata_view._init_as_actual(new)
