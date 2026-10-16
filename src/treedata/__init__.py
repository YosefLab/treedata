from importlib.metadata import version

from ._core.merge import concat
from ._core.read import read_h5ad, read_h5td, read_tree_elems, read_zarr
from ._core.treedata import TreeData
from ._core.write import write_tree_elems

# Submodules need to be imported last
from . import io  # isort: skip

__version__ = version("treedata")
