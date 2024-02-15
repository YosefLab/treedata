"""Code for merging/concatenating TreeData objects."""

from __future__ import annotations

import typing
from collections.abc import (
    Callable,
    Collection,
)
from typing import Any, Literal

from .treedata import TreeData

StrategiesLiteral = Literal["same", "unique", "first", "only"]


def concat(
    tdatas: Collection[TreeData] | typing.Mapping[str, TreeData],
    *,
    axis: Literal[0, 1] = 0,
    join: Literal["inner", "outer"] = "inner",
    merge: StrategiesLiteral | Callable | None = None,
    uns_merge: StrategiesLiteral | Callable | None = None,
    label: str | None = None,
    keys: Collection | None = None,
    index_unique: str | None = None,
    fill_value: Any | None = None,
    pairwise: bool = False,
) -> TreeData:
    raise NotImplementedError("Concatenation not yet implemented")
