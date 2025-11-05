# API

```{eval-rst}
.. module:: treedata
```

The central class:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   TreeData
```

## Combining

Combining TreeData objects:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   concat
```

## Read

Reading TreeData objects:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   read_h5td
   read_zarr
```

TreeData can read Zarr stores created with both Zarr v2 and Zarr v3.

## Write

Writing TreeData objects:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   TreeData.write_h5td
   TreeData.write_zarr
```

TreeData writes Zarr stores using the same compatibility strategy as AnnData,
ensuring interoperability with both Zarr v2 and v3 runtimes.
