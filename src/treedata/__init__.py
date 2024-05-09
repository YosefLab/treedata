from importlib.metadata import version

from ._core.merge import concat
from ._core.read import read_h5ad, read_zarr
from ._core.treedata import TreeData

__version__ = version("treedata")
