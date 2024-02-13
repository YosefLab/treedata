"""
Code for merging/ concatenating TreeData objects.
"""
from __future__ import annotations

import typing
from collections import OrderedDict
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    MutableSet,
    Sequence,
)
from functools import partial, reduce, singledispatch
from itertools import repeat
from operator import and_, or_, sub
from typing import Any, Literal, TypeVar
from warnings import warn

import numpy as np
import pandas as pd

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