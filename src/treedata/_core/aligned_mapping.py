from __future__ import annotations

import warnings
from abc import abstractmethod
from collections import abc as cabc
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import reduce
from typing import TYPE_CHECKING

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from treedata._utils import subset_tree

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.raw import Raw

    from treedata._core.treedata import TreeData

    Index1D = slice | int | str | np.int64 | np.ndarray | list[str | int] | pd.Index
    Index = Index1D | tuple[Index1D | slice, Index1D | slice] | sparse.spmatrix | sparse.sparray


class AxisTreesBase(cabc.MutableMapping):
    """Mapping of key to nx.DiGraph aligned to an axis of parent TreeData."""

    @abstractmethod
    def __init__(self):
        """Abstract constructor."""
        self._tree_to_node = defaultdict(set)
        self._node_to_tree = defaultdict(set)
        self._axis = 0
        self._parent = None
        self._dimnames = ("obs", "var")

    def __repr__(self) -> str:
        """String representation of the object."""
        return f"{type(self).__name__} with keys: {', '.join(self.keys())}"

    def _ipython_key_completions_(self) -> list[str]:
        """IPython key completions."""
        return list(self.keys())

    def _validate_tree(self, tree: nx.DiGraph, key: str) -> set[str]:
        """Validate that graph is a tree and check for overlaps."""
        # Check value type
        if not isinstance(tree, nx.DiGraph):
            raise ValueError(f"Value for key {key} must be a nx.DiGraph")
        # Empty tree
        if tree.number_of_nodes() == 0:
            return set()
        # Check tree topology
        if not nx.is_tree(tree):
            raise ValueError(f"Value for key {key} must be a tree")
        # Check alignment
        nodes = set(tree.nodes)
        if self.parent.alignment == "leaves":
            nodes = {node for node in nodes if tree.out_degree(node) == 0}
        if self.parent.alignment in ["leaves", "nodes"]:
            if not nodes.issubset(self.dim_names):
                raise ValueError(
                    f"Names of {self.parent.alignment} must be in {self.dim}_names when alignment='{self.parent.alignment}'"
                )
        # Check overlap
        if not self.parent.allow_overlap:
            if key in self._tree_to_node:
                new_nodes = nodes.difference(self._tree_to_node[key])
            else:
                new_nodes = nodes
            if self._check_tree_overlap(new_nodes):
                raise ValueError(
                    "Leaf names overlap with leaf names of other trees. Set `allow_overlap=True` to allow this."
                )
        return nodes

    def _check_tree_overlap(self, nodes=()) -> bool:
        """Check if the leaves overlap with other trees."""
        if not nodes:
            return any(len(tree_keys) > 1 for tree_keys in self._node_to_tree.values())
        return any(node in self._node_to_tree for node in nodes)

    def _update_tree_labels(self):
        """Update the tree labels in the parent object."""
        if self.parent._tree_label is not None:
            if self.parent.allow_overlap:
                labels = {k: ",".join(map(str, sorted(v))) for k, v in self._node_to_tree.items()}
            else:
                labels = {k: next(iter(v)) for k, v in self._node_to_tree.items()}
            getattr(self.parent, self.dim)[self.parent._tree_label] = getattr(self.parent, f"{self.dim}_names").map(
                labels
            )

    def _check_uniqueness(self):
        """Check if the names of the axis are unique."""
        names = "Observation" if self.dim == "obs" else "Variable"
        if not getattr(self.parent, self.dim).index.is_unique:
            warnings.warn(
                f"{names} names must be unique to store a tree. Calling `.{self.dim}_names_make_unique` to make them unique.",
                stacklevel=2,
            )
            getattr(self.parent, self.dim).index = ad.utils.make_index_unique(getattr(self.parent, self.dim).index)

    def copy(self) -> AxisTrees:
        """Returns a deep copy of the object."""
        d = AxisTrees(self.parent, self._axis)
        for k, v in self.items():
            d[k] = v.copy()
        return d

    def _view(self, parent: TreeData, subset_idx: Index1D | None) -> AxisTreesView:
        """Returns a subset copy-on-write view of the object."""
        return AxisTreesView(self, parent, subset_idx)

    @property
    def parent(self) -> AnnData | Raw:
        """Parent object of the mapping."""
        return self._parent

    @property
    def attrname(self) -> str:
        """Name of the attribute this is aligned to."""
        return f"{self.dim}t"

    @property
    def axes(self) -> tuple[int]:
        """Axes of the parent this is aligned to"""
        return (self._axis,)

    @property
    def dim(self) -> str:
        """Name of the dimension this aligned to."""
        return self._dimnames[self._axis]

    @property
    def dim_names(self) -> pd.Index:
        """Names of the dimension this aligned to."""
        return (self.parent.obs_names, self.parent.var_names)[self._axis]


class AxisTrees(AxisTreesBase):
    """Mapping of key to nx.DiGraph aligned to an axis of parent TreeData."""

    def __init__(
        self,
        parent: TreeData,
        axis: int,
        vals: Mapping | None = None,
    ):
        self._parent = parent
        self._dimnames = ("obs", "var")
        if axis not in (0, 1):
            raise ValueError()
        self._axis = axis
        self._data = {}
        self._tree_to_node = defaultdict(set)
        self._node_to_tree = defaultdict(set)
        if vals is not None:
            self.update(vals)

    def __getitem__(self, key: str) -> nx.DiGraph:
        """Get item from the mapping."""
        return nx.graphviews.generic_graph_view(self._data[key])

    def __setitem__(self, key: str, value: nx.DiGraph):
        """Set item in the mapping."""
        self._check_uniqueness()
        nodes = self._validate_tree(value, key)

        if key in self._tree_to_node:
            for node in self._tree_to_node[key]:
                self._node_to_tree[node].discard(key)
                if not self._node_to_tree[node]:
                    del self._node_to_tree[node]

        for node in nodes:
            self._node_to_tree[node].add(key)
        self._tree_to_node[key] = nodes

        if not self.parent.is_view:
            self._update_tree_labels()

        self._data[key] = value.copy()
        self.parent._update_has_overlap()

    def __delitem__(self, key: str):
        """Delete item from the mapping."""
        for leaf in self._tree_to_node[key]:
            self._node_to_tree[leaf].remove(key)
            if not self._node_to_tree[leaf]:
                del self._node_to_tree[leaf]
        del self._tree_to_node[key]

        self._update_tree_labels()

        del self._data[key]
        self.parent._update_has_overlap()

    def __len__(self) -> int:
        """Get length of the mapping."""
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        """Check if the mapping contains a key."""
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the mapping."""
        return iter(self._data)

    def _validate_mapping(self) -> bool:
        """Check that the mapping is valid."""
        if self.parent.alignment in ["subset"]:
            return True
        for key, value in self.items():
            try:
                self._validate_tree(value, key)
            except ValueError:
                return False
        return True


class AxisTreesView(AxisTreesBase):
    """View of AxisTree object."""

    def __init__(
        self,
        parent_trees: AxisTreesBase,
        parent_view: TreeData,
        subset_idx: Index1D | None,
    ):
        self.parent_trees = parent_trees
        self._parent = parent_view
        self._dimnames = ("obs", "var")
        self.subset_idx = subset_idx
        self._axis = parent_trees._axis
        self._tree_to_node = parent_trees._tree_to_node
        self._node_to_tree = parent_trees._node_to_tree

    def __getitem__(self, key: str) -> nx.DiGraph:
        """Get item from the mapping."""
        if self.parent.alignment in ["leaves", "nodes"]:
            nodes = self.parent_trees._tree_to_node[key]
            subset_nodes = nodes.intersection(self.dim_names.values)
            tree_view = subset_tree(self.parent_trees[key], subset_nodes, asview=True, alignment=self.parent.alignment)
        else:
            tree_view = nx.graphviews.generic_graph_view(self.parent_trees[key])
        return tree_view

    def __setitem__(self, key: str, value: nx.DiGraph):
        """Set item in the mapping."""
        self._check_uniqueness()
        self._validate_tree(value, key)
        warnings.warn(
            f"Setting element `.{self.attrname}['{key}']` of view, initializing view as actual.", stacklevel=2
        )
        with view_update(self.parent, self.attrname, ()) as new_mapping:
            new_mapping[key] = value

    def __delitem__(self, key: str):
        """Delete item from the mapping."""
        if key not in self:
            raise KeyError(
                "'{key!r}' not found in view of {self.attrname}"
            )  # Make sure it exists before bothering with a copy
        warnings.warn(
            f"Removing element `.{self.attrname}['{key}']` of view, initializing view as actual.", stacklevel=2
        )
        with view_update(self.parent, self.attrname, ()) as new_mapping:
            del new_mapping[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the mapping."""
        return iter(self.parent_trees)

    def __contains__(self, key: str) -> bool:
        """Check if the mapping contains a key."""
        return key in self.parent_trees

    def __len__(self) -> int:
        """Get length of the mapping."""
        return len(self.parent_trees)

    def _validate_mapping(self) -> bool:
        """Check that the mapping is valid."""
        if self.parent.alignment in ["leaves", "subset"]:
            return True
        for key in self.keys():
            try:
                self[key]
            except ValueError:
                return False
        return True


@contextmanager
def view_update(tdata_view: TreeData, attr_name: str, keys: tuple[str, ...]) -> Iterator:
    """Context manager for updating a view of an TreeData object.

    Contains logic for "actualizing" a view. Yields the object to be modified in-place.

    Parameters
    ----------
    tree_view
        A view of a TreeData object
    attr_name
        Name of the attribute being updated
    keys
        Keys to the attribute being updated

    Yields
    ------
    `adata.attr[key1][key2][keyn]...`
    """
    new = tdata_view.copy()
    attr = getattr(new, attr_name)
    container = reduce(lambda d, k: d[k], keys, attr)
    yield container
    tdata_view._init_as_actual(new)
