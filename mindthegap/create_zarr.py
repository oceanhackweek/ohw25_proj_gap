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

from os import path
from pathlib import Path
from typing import Union
import numpy as np
import dask.array as da
import xarray as xr

def data_preprocessing(
    zarr_ds,
    features,
    train_year,
    train_range,
    zarr_tag="full_2days",
    datadir: Union[str, Path] = "data",
):
    numer_features = []  # numerical features
    cat_features = []    # categorical features
    zarr_label = f"{train_year}_{train_range}_{zarr_tag}"

    datadir = Path(datadir)
    store_dir = datadir / f"{zarr_label}.zarr"
    if store_dir.exists():
        print("Zarr file exists")
        return zarr_label

    print("label created")

    # --- raw data features ---
    for f in features:
        numer_features.append(zarr_ds[f].data)  # keep as dask-backed arrays
    print("raw data features added")

    # --- label (log CHL) ---
    CHL_data = zarr_ds["CHL_cmes-level3"].data
    CHL_data = da.log(CHL_data)  # keep in dask
    print("CHL logged")

    # --- seasonal sin/cos of day ---
    time_data = zarr_ds.time.data  # dask or numpy datetime64 array
    day_rad = ((time_data - np.datetime64("1900-01-01")) / np.timedelta64(1, "D")) / 365 * 2 * np.pi
    day_rad = day_rad.astype(np.float32)

    # expand to (time, lat, lon) by broadcasting, then rechunk later
    day_sin = da.sin(day_rad)[:, None, None]
    day_cos = da.cos(day_rad)[:, None, None]

    numer_features.append(day_sin)
    numer_features.append(day_cos)
    print("sin/cos time added")

    # --- artificially masked CHL (10-day shift) ---
    cloud = zarr_ds["CHL_cmes-cloud"].data  # dask array (time, lat, lon)
    # shift by 10 days along time without changing chunk topology too much
    day_shift_flag = da.roll(cloud, shift=-10, axis=0)
    masked_CHL = da.where(day_shift_flag == 0, np.nan, CHL_data)
    numer_features.append(masked_CHL)
    print("masked CHL added")

    # --- prev/next day CHL ---
    t, y, x = CHL_data.shape
    zero_slice = da.zeros((1, y, x), dtype=CHL_data.dtype)
    prev_day = da.concatenate([zero_slice, CHL_data[:-1]], axis=0)
    next_day = da.concatenate([CHL_data[1:], zero_slice], axis=0)
    numer_features.append(prev_day)
    numer_features.append(next_day)
    print("prev/next day CHL added")

    # --- categorical flags (one-hot) ---
    land_flag = da.where(cloud[0] == 2, 1, 0).astype(np.int8)
    land_flag = da.broadcast_to(land_flag, CHL_data.shape)  # (t, y, x)
    cat_features.append(land_flag)

    real_cloud_flag = da.where(cloud == 1, 1, 0).astype(np.int8)
    cat_features.append(real_cloud_flag)

    valid_CHL_flag = da.where(~da.isnan(masked_CHL), 1, 0).astype(np.int8)
    cat_features.append(valid_CHL_flag)

    fake_cloud_flag = da.where((land_flag + real_cloud_flag + valid_CHL_flag) == 0, 1, 0).astype(np.int8)
    cat_features.append(fake_cloud_flag)
    print("flags added")

    # --- training window indices ---
    train_start_ind = np.where(zarr_ds.time.values == np.datetime64(f"{train_year}-01-01"))[0][0]
    train_end_ind   = np.where(zarr_ds.time.values == np.datetime64(f"{train_year + train_range}-01-01"))[0][0]

    # --- stats for numericals (over training window) ---
    feat_mean, feat_stdev = [], []
    for feature in numer_features:
        f_train = feature[train_start_ind:train_end_ind]
        feat_mean.append(da.nanmean(f_train).compute())
        feat_stdev.append(da.nanstd(f_train).compute())
    print("means/stds computed")

    # --- standardize numericals ---
    numer_features_stdized = []
    feature_shape = (t, y, x)
    for feature, mean, stdev in zip(numer_features, feat_mean, feat_stdev):
        numer_features_stdized.append((feature - mean) / stdev)
    print("numericals standardized")

    # --- CHL standardization (global over all time here) ---
    CHL_mean = da.nanmean(CHL_data).compute()
    CHL_stdev = da.nanstd(CHL_data).compute()
    np.save(
        datadir / f"{zarr_label}.npy",
        {"CHL": np.array([CHL_mean, CHL_stdev]),
         "masked_CHL": np.array([feat_mean[-3], feat_stdev[-3]])},
        allow_pickle=True,  # needed to save dicts with np.save
    )
    CHL_data_stdized = (CHL_data - CHL_mean) / CHL_stdev
    print("label standardized")

    # make chunks UNIFORM
    T = 100                 # time chunk
    Y = y                   # one chunk across latitude
    X = x                   # one chunk across longitude

    def ensure_chunks(arr):
        # arr is a dask array shaped (t,y,x)
        return da.rechunk(arr, chunks=(T, Y, X))

    numer_features_stdized = [ensure_chunks(a) for a in numer_features_stdized]
    cat_features           = [ensure_chunks(a) for a in cat_features]
    CHL_data_stdized       = ensure_chunks(CHL_data_stdized)

    numer_var_names = list(features) + ["sin_time", "cos_time", "masked_CHL", "prev_day_CHL", "next_day_CHL"]
    cat_var_names   = ["land_flag", "real_cloud_flag", "valid_CHL_flag", "fake_cloud_flag"]

    print("creating zarr")
    create_zarr(
        zarr_ds,
        numer_features_stdized,
        numer_var_names,
        cat_features,
        cat_var_names,
        CHL_data_stdized,   # keep as dask to write lazily
        zarr_label,
        datadir,
    )

    # clean up (optional)
    del time_data, day_rad, day_sin, day_cos
    del numer_features, numer_features_stdized, numer_var_names
    del cat_features, cat_var_names, CHL_data, CHL_data_stdized
    del feat_mean, feat_stdev, prev_day, next_day, masked_CHL, day_shift_flag
    return zarr_label
