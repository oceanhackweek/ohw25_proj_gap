from pathlib import Path
from typing import Sequence, Union
import numpy as np
import xarray as xr


def create_zarr(
    zarr_ds: xr.Dataset,
    numer_features: Sequence[np.ndarray],
    numer_var_names: Sequence[str],
    cat_features: Sequence[np.ndarray],
    cat_var_names: Sequence[str],
    CHL_data: np.ndarray,
    zarr_label: str,
    datadir: Union[str, Path] = "data",
):
    """
    Create a feature/target dataset and write it to a Zarr store.

    This function packages the provided numerical and categorical feature arrays
    (e.g., standardized predictors) together with the target variable
    (standardized/log CHL) into a single ``xarray.Dataset`` with coordinates
    derived from ``zarr_ds`` and writes it to a Zarr store on disk.

    Parameters
    ----------
    zarr_ds : xarray.Dataset
        A reference dataset used to supply coordinates and dimension ordering.
        Must contain coordinates named ``time``, ``lat``, and ``lon`` that match
        the shapes of the provided feature and target arrays.
    numer_features : Sequence[np.ndarray]
        Iterable of 3D arrays for numerical predictors, each shaped
        ``(time, lat, lon)`` and aligned to ``zarr_ds``.
    numer_var_names : Sequence[str]
        Variable names corresponding 1:1 to ``numer_features`` (same length).
    cat_features : Sequence[np.ndarray]
        Iterable of 3D arrays for categorical/flag predictors, each shaped
        ``(time, lat, lon)`` and aligned to ``zarr_ds``.
    cat_var_names : Sequence[str]
        Variable names corresponding 1:1 to ``cat_features`` (same length).
    CHL_data : np.ndarray
        Target variable array shaped ``(time, lat, lon)`` (e.g., standardized or
        log-transformed CHL), aligned to ``zarr_ds``.
    zarr_label : str
        Label used to name the Zarr store (e.g., ``"<label>.zarr"``).
    datadir : str or path-like, optional
        Directory where the Zarr store will be created. Defaults to ``"data"``.
        (This is the only addition versus the original function.)

    Returns
    -------
    str
        The filesystem path to the written Zarr store.

    Notes
    -----
    - Assumes all arrays share the same dimensions and ordering:
      ``(time, lat, lon)``.
    - Variable names should be valid xarray variable identifiers. If any name
      contains characters that are awkward to index (e.g., hyphens), prefer
      selection via ``.data_vars["name-with-hyphen"]``.
    - This version preserves the original computational logic; only the output
      directory is now configurable via ``datadir``.

    Side Effects
    ------------
    - Writes a Zarr store to ``<datadir>/<zarr_label>.zarr`` (overwrites if
      it already exists).
    """
    # --- Build the dataset exactly as in the original, only changing the output path ---
    # Use coords from the reference dataset to guarantee consistency
    coords = {}
    for c in ("time", "lat", "lon"):
        if c in zarr_ds.coords:
            coords[c] = zarr_ds.coords[c]
        elif c in zarr_ds:
            coords[c] = zarr_ds[c]
        else:
            raise ValueError(f"Required coordinate '{c}' not found in zarr_ds.")

    # Map features to names
    if len(numer_features) != len(numer_var_names):
        raise ValueError("Length mismatch: numer_features vs numer_var_names.")
    if len(cat_features) != len(cat_var_names):
        raise ValueError("Length mismatch: cat_features vs cat_var_names.")

    data_vars = {}

    # Numerical predictors
    for name, arr in zip(numer_var_names, numer_features):
        data_vars[name] = (("time", "lat", "lon"), arr)

    # Categorical/flag predictors
    for name, arr in zip(cat_var_names, cat_features):
        data_vars[name] = (("time", "lat", "lon"), arr)

    # Target variable
    data_vars["CHL"] = (("time", "lat", "lon"), CHL_data)

    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Ensure output directory exists
    datadir = Path(datadir)
    datadir.mkdir(parents=True, exist_ok=True)

    # Write Zarr (same behavior, only path is configurable now)
    store_path = datadir / f"{zarr_label}.zarr"
    ds_out.to_zarr(store_path.as_posix(), mode="w", consolidated=True)

    return store_path.as_posix()