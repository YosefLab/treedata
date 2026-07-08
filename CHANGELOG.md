# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

### Changed

### Fixed

## [0.3.1] - 2026-07-08

### Added

### Changed

- Require `anndata>=0.13` (and consequently Python `>=3.12`, which anndata 0.13 requires) (#88)

### Fixed

- `TreeData.__getitem__` now delegates `anndata.acc` accessor indexers (`AdRef`, `MapAcc`, `RefAcc`) to `AnnData.__getitem__` instead of routing them through `_normalize_indices`. This fixes an `IndexError: Unknown indexer` from `obs_vector`/`var_vector` (and any `var_names`/layer column lookup) under anndata ≥ 0.13 (#88)

## [0.3.0] - 2026-07-07

### Added

### Changed

- Require `zarr>=3` (anndata 0.13 no longer supports zarr <3). Removed the dead `zarr<3` (`ZARR_V2`) code branches and the dead `anndata<0.11` (`USE_EXPERIMENTAL`) code branches now that the supported floors exceed them  (#88)
- Zarr writes now store `obst`/`vart` in a columnar layout (one array per node/edge attribute key) instead of a single JSON scalar. Dense numeric/boolean columns use native dtypes; ragged or complex columns fall back to per-element JSON. This fixes a `TypeError: string too large to store inside array` crash when writing large trees to zarr v3 (#86)
- `dtype` parameter in `TreeData.__init__` is now deprecated; passing it raises a `FutureWarning` and has no effect, matching anndata's removal of this argument
- `obsm`, `varm`, `layers`, `raw`, `shape`, `filename`, `filemode`, `asview`, and related parameters in `TreeData.__init__` are now keyword-only, matching the anndata 0.13.x API
- Added compatibility with anndata 0.13.x: removed `dtype` from internal `_init_as_actual` call, which was removed from AnnData in 0.13.x
- Added compatibility with anndata 0.13.x "Unify X and layers" change: `layers.copy()` and `dict(layers)` no longer include the internal `None` key (used to store X) in `_mutated_copy` and write routines

### Fixed

- Fixed `UserWarning: Duplicate name: 'zarr.json'` spam when writing to a zarr `ZipStore` by staging the write in a `MemoryStore` and copying to the `ZipStore` once (#86)
- `read_zarr` now auto-detects `.zip` paths and opens them as a `ZipStore` (#86)
- Node/edge attributes explicitly set to `None` are now preserved through a zarr round-trip and kept distinct from absent attributes (#86)
- Rewriting into an existing zarr group now removes trees that are no longer present, so the persisted tree set matches the `TreeData` being written (#86)

## [0.2.5] - 2026-04-20

### Added

### Changed

- Updated anndata dependency (#75)

### Fixed

- Removed pathlib dependency (#83)

## [0.2.4] - 2025-11-05

### Added

### Changed

- Deprecated `TreeData.obst_keys` and `TreeData.vart_keys` for consistency with AnnData (https://github.com/scverse/anndata/pull/2093) (#73)

### Fixed

- Added support for zarr v3 (#73)
- Eliminated deprecations warnings from AnnData>=0.12.0 (#73)

## [0.2.3] - 2025-10-14

### Added

### Changed

- Optimized tree overlap detection to speed up copying and subsetting with many trees (#66)
- Switch to hatch with v0.0.6 template update (#61)

### Fixed

- Fixed codecov configuration (#64)

## [0.2.2] - 2025-09-18

### Added

-  `tdata.has_overlap` parameter to check whether the `TreeData` object contains overlapping trees (#59)

### Changed

- Default value for `tdata.allow_overlap` is now `True` (#60)

### Fixed

## [0.2.1] - 2025-07-10

### Added

-  support for instantiating `TreeData` objects with only the tree structure (#56)

### Changed

- Updated docs to clarify all the ways `TreeData` can be instantiated (#56)

### Fixed

## [0.2.0] - 2025-06-16

### Added

- `alignment` parameter which allows for `obs_names` and `var_names` aligned to either the leaves, nodes, or a subset of leaves and nodes in trees stored in the `obst` and `vart` fields. Added a tutorial describing how `alignment` works (#55)

### Changed

- `read_h5ad` and `write_h5ad` and been renamed `read_h5td` and `write_h5td` to clarify that the `treedata` format differs from `anndata`. `read_h5ad` and `write_h5ad` will be removed in `v1.0.0` (#56)

### Fixed

- Fixed typing issues (#51)

## [0.1.3] - 2025-01-20

### Added

### Changed

### Fixed

- Fixed typing issues (#51)

- Fixed `ImportError: zarr-python major version > 2 is not supported'` error with Python 12 (#46)

## [0.1.2] - 2024-12-02

### Added

### Changed

### Fixed

- Fixed `KeyError: "Unable to synchronously open object (object 'X' doesn't exist)"'` when reading h5ad without X field (#40)

## [0.1.1] - 2024-11-25

### Added

- Axis in `td.concat` can now be specified with `obs` and `var` (#40)

### Changed

### Fixed

- Fixed `ImportError: cannot import name '_resolve_dim' from 'anndata._core.merge'` caused by anndata update (#40)

## [0.1.0] - 2024-09-27

### Added

### Changed

- Encoding of `treedata` attributes in h5ad and zarr files. `label`, `allow_overlap`, `obst`, and `vart` are now separate fields in the file. (#31)

### Fixed

- `TreeData` objects with `.raw` specified can now be read (#31)

## [0.0.4] - 2024-09-02

### Added

### Changed

### Fixed

- Fixed typing bug introduced by anndata update (#29)

## [0.0.3] - 2024-08-21

### Added

- Add concatenation tutorial to documentation (#27)

### Changed

- `obst` and `vart` create local copy of `nx.DiGraphs` that are added (#26)
- `TreeData.label` value remains the same after `td.concat` as long as all `label` values are the same for all objects (#27)

### Fixed

- Fixed bug which caused key to be listed twice in `label` column after value update in `obst` or `vart` (#26)

## [0.0.2] - 2024-06-18

### Changed

- Empty trees are now allowed to avoid error on subsetting (#13)
- How trees are stored in h5ad and zarr files (#16)
- Format of label column with multiple trees ([1,2] -> 1,2) (#16)

### Fixed

- Fixed issue with slow read/write of large trees

## [0.0.1] - 2024-05-13

### Added

- TreeData class for storing and manipulating trees
- Read/write trees to h5ad and zarr files
- Concatenate trees with similar API to AnnData
