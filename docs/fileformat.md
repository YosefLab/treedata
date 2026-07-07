# On-disk format

TreeData is written to `.h5td` (HDF5) and `.zarr` stores using the same
hierarchical, self-describing layout as AnnData. A TreeData file **is** an
AnnData file with a few extra top-level elements, so almost everything is
covered by the [AnnData on-disk format][anndata-format] — read that first. This
page documents only what TreeData adds.

[anndata-format]: https://anndata.scverse.org/en/stable/fileformat-prose.html

## Top-level

The root group carries `encoding-type: "treedata"` and, in addition to the
standard AnnData elements (`X`, `obs`, `var`, `obsm`, `uns`, …), stores:

| Element | Type | Description |
| --- | --- | --- |
| `label` | scalar (string) or absent | Column in `obs`/`var` that holds each node's tree key |
| `allow_overlap` | scalar (bool) | Whether trees may share nodes |
| `alignment` | scalar (string) | One of `leaves`, `nodes`, `subset` |
| `obst` | see below | Trees aligned to observations |
| `vart` | see below | Trees aligned to variables |

`obst` and `vart` are mappings of a key to a :class:`networkx.DiGraph`. They are
the only elements stored differently from AnnData, and the layout depends on the
backend.

## Trees in HDF5 (`.h5td`)

Each of `obst`/`vart` is stored as a **single JSON string** dataset mapping tree
key → `{"nodes": {...}, "edges": {...}}`, with node and edge attributes inline:

```json
{"tree1": {"nodes": {"root": {"depth": 0}, "0": {}},
           "edges": {"root": {"0": {"length": 1.5}}}}}
```

## Trees in zarr

zarr cannot store a single arbitrarily large string, so each of `obst`/`vart` is
a **group** with one subgroup per tree, stored columnar:

```
obst/
  tree1/
    nodes                 # 1-D array of node IDs
    edges                 # (n_edges, 2) array of (source, target) pairs
    node_attrs/           # one array per node-attribute key (optional)
      depth
      characters
    edge_attrs/           # one array per edge-attribute key (optional)
      length
```

The number of arrays scales with the number of **attribute keys**, not the
number of nodes.

Each attribute column has one entry per node (or edge), in the order of `nodes`
(or `edges`), and is encoded one of two ways:

- **Native** — a dense column whose values are all a single numeric or boolean
  type is stored with the corresponding native array dtype (e.g. `int64`,
  `float64`, `bool`).
- **JSON** — any other column (strings, ragged/nested values, mixed types,
  columns missing from some nodes) is stored as a string array of per-element
  JSON. The empty string `""` marks an absent attribute, so a value explicitly
  set to `null` is preserved and kept distinct from a missing one.

:::{note}
zarr files written before treedata 0.3.0 stored `obst`/`vart` as a single JSON
string (as in HDF5). Such files are still read correctly; the format is detected
at read time.
:::
