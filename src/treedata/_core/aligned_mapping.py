from __future__ import annotations

from collections import abc as cabc
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import (
    TYPE_CHECKING,
    Literal,
)

import anndata as ad
import networkx as nx
import pandas as pd

from treedata._utils import get_leaves

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.raw import Raw


class AxisTrees(cabc.MutableMapping):
    """Mapping of key to nx.DiGraph aligned to an axis of parent AnnData."""

    def __init__(
        self,
        parent: ad.AnnData,
        axis: int,
        vals: Mapping | None = None,
    ):
        self._parent = parent
        if axis not in (0, 1):
            raise ValueError()
        self._axis = axis
        self._data = {}
        self._membership = defaultdict(list)
        if vals is not None:
            self.update(vals)

        # consider storing tree membership for each observation

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
        # Check overlap
        if not self.parent._allow_tree_overlap:
            if set(leaves).intersection(self._membership.keys()):
                raise ValueError(f"Leaf nodes of tree for key {key} overlap with other trees")
        return value

    def _update_tree_labels(self):
        if self.parent._tree_label is not None:
            if self.parent._allow_tree_overlap:
                mapping = self._membership
            else:
                mapping = {k: v[0] for k, v in self._membership.items()}
            self.parent.obs[self.parent._tree_label] = self.parent.obs_names.map(mapping)

    def copy(self):
        d = self._actual_class(self.parent, self._axis)
        for k, v in self.items():
            d[k] = v.copy()
        return d

    def __getitem__(self, key: str) -> nx.DiGraph:
        return self._data[key]

    def __setitem__(self, key: str, value: nx.DiGraph):
        value = self._validate_value(value, key)

        leaves = get_leaves(value)
        for leaf in leaves:
            self._membership[leaf].append(key)

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
