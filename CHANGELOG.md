# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.0.4] - 2024-09-02

### Added

### Changed

### Fixed

-   Fixed typing bug introduced by anndata update (#29)

## [0.0.3] - 2024-08-21

### Added

-   Add concatenation tutorial to documentation (#27)

### Changed

-   `obst` and `vart` create local copy of `nx.DiGraphs` that are added (#26)
-   `TreeData.label` value remains the same after `td.concat` as long as all `label` values are the same for all objects (#27)

### Fixed

-   Fixed bug which caused key to be listed twice in `label` column after value update in `obst` or `vart` (#26)

## [0.0.2] - 2024-06-18

### Changed

-   Empty trees are now allowed to avoid error on subsetting (#13)
-   How trees are stored in h5ad and zarr files (#16)
-   Format of label column with multiple trees ([1,2] -> 1,2) (#16)

### Fixed

-   Fixed issue with slow read/write of large trees

## [0.0.1] - 2024-05-13

### Added

-   TreeData class for storing and manipulating trees
-   Read/write trees to h5ad and zarr files
-   Concatenate trees with similar API to AnnData
