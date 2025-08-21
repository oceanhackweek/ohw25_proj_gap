from typing import Union
import xarray as xr

def crop_to_multiple(
    ds: Union[xr.Dataset, xr.DataArray],
    lat: str = "lat",
    lon: str = "lon",
    multiple: int = 8,
    center: bool = False,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Crop an xarray Dataset or DataArray along latitude/longitude so that the
    spatial shape is divisible by `multiple` (useful for U-Net down/upsampling).

    This is a *view*-like operation (no data copy) that trims rows/columns from
    the edges only. It does not pad. If `center=True`, the function crops
    symmetrically; otherwise it drops only from the end.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input object with spatial dims (`lat`, `lon` by default).
    lat : str, default "lat"
        Name of the latitude dimension to crop.
    lon : str, default "lon"
        Name of the longitude dimension to crop.
    multiple : int, default 8
        Target multiple for both spatial dimensions. For a U-Net with `D`
        pooling levels, use `multiple = 2**D` (e.g., D=3 → 8).
    center : bool, default False
        If True, crop symmetrically (half from the start, half from the end).
        If False, drop only from the end (keeps the origin intact).

    Returns
    -------
    xr.Dataset or xr.DataArray
        Cropped object (same type as input) whose spatial shape is divisible by `multiple`.

    Raises
    ------
    KeyError
        If `lat` or `lon` dims are not present in `ds`.

    Notes
    -----
    - If a dimension is already divisible by `multiple`, it is left unchanged.
    - If `ds.sizes[lat] < multiple` (or same for `lon`), this will crop to zero
      for that dimension; consider padding instead in that case (e.g., `xr.pad`).
    - Coordinates and attributes are preserved by `isel`.

    Examples
    --------
    Basic use with a Dataset:
    >>> ds_aligned = crop_to_multiple(zarr_ds, multiple=8)
    >>> ds_aligned.sizes["lat"], ds_aligned.sizes["lon"]
    (104, 152)  # for an original 105x153

    Symmetric crop (centered):
    >>> ds_centered = crop_to_multiple(zarr_ds, multiple=8, center=True)

    With a DataArray:
    >>> chl = zarr_ds["CHL_cmes-level3"]
    >>> chl_aligned = crop_to_multiple(chl, multiple=8)

    Using U-Net depth to choose the multiple:
    >>> depth = 3
    >>> m = 2 ** depth
    >>> ds_aligned = crop_to_multiple(zarr_ds, multiple=m)
    """
    # Validate required dims
    if lat not in ds.dims or lon not in ds.dims:
        missing = [d for d in (lat, lon) if d not in ds.dims]
        raise KeyError(f"Missing required dimension(s): {missing}. Present: {list(ds.dims)}")

    nlat = ds.sizes[lat]
    nlon = ds.sizes[lon]
    rlat = nlat % multiple
    rlon = nlon % multiple

    # Already aligned → return as-is
    if rlat == 0 and rlon == 0:
        return ds

    if not center:
        # Drop only from the end so indices/geo origin are preserved
        sl_lat = slice(0, nlat - rlat) if rlat else slice(0, nlat)
        sl_lon = slice(0, nlon - rlon) if rlon else slice(0, nlon)
    else:
        # Symmetric crop: split the remainder on both sides
        lat_left = rlat // 2
        lat_right = rlat - lat_left
        lon_left = rlon // 2
        lon_right = rlon - lon_left
        sl_lat = slice(lat_left, nlat - lat_right)
        sl_lon = slice(lon_left, nlon - lon_right)

    # isel preserves coords/attrs and is lazy for dask-backed arrays
    return ds.isel({lat: sl_lat, lon: sl_lon})

import numpy as np

# Helper functions in mindthegap
# - `unstdize`: unstandardize model outputs back to the original scale
# - `compute_mae`: mean absolute error, ignoring NaNs
# - `compute_mse`: mean squared error, ignoring NaNs


def unstdize(stdized_image, mean, stdev):
    """
    Unstandardize an array from standardized units back to the original scale.

    Given values standardized as (x - mean) / stdev, this function inverts the
    transform to recover x.

    Parameters
    ----------
    stdized_image : array-like
        Standardized values (e.g., model outputs). Can be a NumPy array or
        any array-like object broadcastable with `mean` and `stdev`.
    mean : float or array-like
        Mean used during standardization. May be a scalar or an array
        broadcastable to `stdized_image`.
    stdev : float or array-like
        Standard deviation used during standardization. May be a scalar or an array
        broadcastable to `stdized_image`.

    Returns
    -------
    array-like
        Unstandardized values on the original scale.

    Examples
    --------
    >>> y_std = np.array([0.0, 1.0, -1.0])
    >>> unstdize(y_std, mean=10.0, stdev=2.0)
    array([10., 12.,  8.])

    >>> y_std = np.array([[0., 1.], [np.nan, -0.5]])
    >>> unstdize(y_std, mean=5.0, stdev=2.0)
    array([[5. , 7. ],
           [nan, 4. ]])
    """
    return stdized_image * stdev + mean


def compute_mae(y_true, y_pred):
    """
    Compute mean absolute error (MAE) while ignoring NaN pairs.

    Elements where either `y_true` or `y_pred` is NaN are excluded from the average.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values. Must be the same shape as `y_true`.

    Returns
    -------
    float
        Mean absolute error over valid (non-NaN) pairs.

    Examples
    --------
    >>> yt = np.array([1.0, 2.0, np.nan, 4.0])
    >>> yp = np.array([0.5, 2.5, 3.0, np.nan])
    >>> compute_mae(yt, yp)
    0.5

    Notes
    -----
    - If all pairs are NaN, `np.mean([])` will return `nan`.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def compute_mse(y_true, y_pred):
    """
    Compute mean squared error (MSE) while ignoring NaN pairs.

    Elements where either `y_true` or `y_pred` is NaN are excluded from the average.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values. Must be the same shape as `y_true`.

    Returns
    -------
    float
        Mean squared error over valid (non-NaN) pairs.

    Examples
    --------
    >>> yt = np.array([1.0, 2.0, np.nan, 4.0])
    >>> yp = np.array([0.5, 2.5, 3.0, np.nan])
    >>> compute_mse(yt, yp)
    0.25

    See Also
    --------
    compute_mae : Mean absolute error with the same NaN handling.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)
