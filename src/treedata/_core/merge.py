"""Code for merging/concatenating TreeData objects."""

from __future__ import annotations

import typing
import warnings
from collections.abc import (
    Callable,
    Collection,
)
from functools import reduce
from typing import Any, Literal

import anndata as ad
import pandas as pd
from anndata._core.merge import resolve_merge_strategy

from treedata._utils import _resolve_axis, combine_trees

from .treedata import TreeData

StrategiesLiteral = Literal["same", "unique", "first", "only"]


def concat(
    tdatas: Collection[TreeData] | typing.Mapping[str, TreeData],
    *,
    axis: Literal["obs", 0, "var", 1] = "obs",
    join: Literal["inner", "outer"] = "inner",
    merge: StrategiesLiteral | Callable | None = None,
    uns_merge: StrategiesLiteral | Callable | None = None,
    label: str | None = None,
    keys: Collection | None = None,
    fill_value: Any | None = None,
    pairwise: bool = False,
) -> TreeData:
    """Concatenates TreeData objects along an axis.

    Parameters
    ----------
    tdatas
        The objects to be concatenated. If a Mapping is passed, keys are used for the `keys`
        argument and values are concatenated.
    axis
        Which axis to concatenate along.
    join
        How to align values when concatenating. If "outer", the union of the other axis
        is taken. If "inner", the intersection.
        for more.
    merge
        How elements not aligned to the axis being concatenated along are selected.
        Currently implemented strategies include:

        * `None`: No elements are kept.
        * `"same"`: Elements that are the same in each of the objects.
        * `"unique"`: Elements for which there is only one possible value.
        * `"first"`: The first element seen at each from each position.
        * `"only"`: Elements that show up in only one of the objects.
    uns_merge
        How the elements of `.uns` are selected. Uses the same set of strategies as
        the `merge` argument, except applied recursively.
    label
        Column in axis annotation (i.e. `.obs` or `.var`) to place batch information in.
        If it's None, no column is added.
    keys
        Names for each object being added. These values are used for column values for
        `label` or appended to the index if `index_unique` is not `None`. Defaults to
        incrementing integer labels.
    fill_value
        When `join="outer"`, this is the value that will be used to fill the introduced
        indices. By default, sparse arrays are padded with zeros, while dense arrays and
        DataFrames are padded with missing values.
    pairwise
        Whether pairwise elements along the concatenated dimension should be included.
        This is False by default, since the resulting arrays are often not meaningful.
    """
    axis, dim = _resolve_axis(axis)
    alt_axis, alt_dim = _resolve_axis(axis=1 - axis)
    merge = resolve_merge_strategy(merge)

    # Check indices
    concat_indices = pd.concat([pd.Series(getattr(t, f"{dim}_names")) for t in tdatas], ignore_index=True)
    if not concat_indices.is_unique:
        raise ValueError(f"{dim}_names must be unique to concatenate along axis {axis}")
    alt_indices = [getattr(t, f"{alt_dim}_names") for t in tdatas]
    if join == "inner":
        alt_indices = reduce(lambda x, y: x.intersection(y), alt_indices)
    else:
        alt_indices = reduce(lambda x, y: x.union(y), alt_indices)

    # Concatenate anndata
    adata = ad.concat(
        tdatas,
        axis=axis,
        join=join,
        merge=merge,
        uns_merge=uns_merge,
        label=label,
        keys=keys,
        index_unique=None,
        fill_value=fill_value,
        pairwise=pairwise,
    )

    # Create new TreeData object
    label = {t.label for t in tdatas if t.label is not None}
    if len(label) > 1:
        warnings.warn("Multiple label values found. Setting to `tree`.", stacklevel=2)
        label = "tree"
    else:
        label = next(iter(label), None)
    allow_overlap = any(t.allow_overlap for t in tdatas)
    tdata = TreeData(adata, allow_overlap=allow_overlap, label=label)

    # Trees for concatenation axis
    concat_trees = [getattr(t, f"{dim}t") for t in tdatas]
    unique_keys = {key for mapping in concat_trees for key in mapping.keys()}
    for key in unique_keys:
        trees = [mapping[key] for mapping in concat_trees if key in mapping]
        tree = combine_trees(trees)
        getattr(tdata, f"{dim}t")[key] = tree

    # Trees for other axis
    if join == "inner" and alt_axis == 0:
        tdatas = [t[alt_indices, :] for t in tdatas]
    elif join == "inner" and alt_axis == 1:
        tdatas = [t[:, alt_indices] for t in tdatas]
    alt_trees = merge([getattr(t, f"{alt_dim}t") for t in tdatas])
    for key, tree in alt_trees.items():
        getattr(tdata, f"{alt_dim}t")[key] = tree

    return tdata
