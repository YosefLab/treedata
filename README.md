[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![PyPI](https://img.shields.io/pypi/v/treedata.svg)](https://pypi.org/project/treedata)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/YosefLab/treedata/test.yaml?branch=main
[link-tests]: https://github.com/YosefLab/treedata/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/treedata

<img
  src="https://raw.githubusercontent.com/YosefLab/treedata/main/docs/_static/img/treedata_schema.svg"
  class="dark-light" align="right" width="350" alt="image"
/>

# TreeData - AnnData with trees

TreeData is a lightweight wrapper around AnnData which adds two additional attributes, `obst` and `vart`, to store [nx.DiGraph] trees for observations and variables. TreeData has the same interface as AnnData and is fully compatible with [scverse] packages like [scanpy].

To learn more about TreeData, please refer to the [documentation][link-docs] or checkout the [getting started guide][link-getting-started].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install treedata:

1. Install the latest release of `treedata` from [PyPI](https://pypi.org/project/treedata):

```bash
pip install treedata
```

2. Install the latest development version:

```bash
pip install git+https://github.com/YosefLab/treedata.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and bug reports please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse]: https://scverse.org/
[scanpy]: https://scanpy.readthedocs.io/
[nx.DiGraph]: https://networkx.org/documentation/stable/reference/classes/digraph.html
[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/YosefLab/treedata/issues
[changelog]: https://treedata.readthedocs.io/en/latest/changelog.html
[link-docs]: https://treedata.readthedocs.io
[link-getting-started]: https://treedata.readthedocs.io/en/latest/notebooks/getting-started.html
[link-api]: https://treedata.readthedocs.io/latest/api.html
