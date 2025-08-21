from .utils import unstdize, compute_mae, compute_mse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Sequence, Union

def plot_prediction_observed(
    zarr_stdized,
    zarr_label, 
    model, 
    date_to_predict,
    datadir: Union[str, Path] = "data",
):
    """
    Plot observed vs. predicted log(Chl-a) for a single date, along with flags and differences.

    This function:
      1) loads mean/std used during standardization from ``{datadir}/{zarr_label}.npy``,
      2) builds a predictor tensor X for the requested date from all variables in ``zarr_stdized``
         except ``"CHL"``, filling NaNs with 0.0,
      3) gets the observed Level-3 log(Chl-a) for the same date from the global ``zarr_ds``,
      4) runs the model to produce standardized predictions, then unstandardizes back to log-scale
         using ``utils.unstdize``,
      5) masks predictions where the observation is NaN, and
      6) produces a 2×2 panel plot: observed, flags, predicted, and (observed − predicted).

    Parameters
    ----------
    zarr_stdized : xarray.Dataset
        Dataset containing standardized predictors (and flags) for the model input. Must include
        the variables used by the model as well as ``'CHL'`` (which is removed from X).
    zarr_label : str
        Label used to locate the standardization sidecar file ``{datadir}/{zarr_label}.npy``
        containing ``{'CHL': array([mean, std]), 'masked_CHL': ...}``.
    model : tf.keras.Model
        Trained U-Net (or compatible) model expecting input shaped (H, W, C) and returning
        a single-channel prediction of standardized log(Chl-a).
    date_to_predict : str or numpy.datetime64 or pandas.Timestamp
        Date to visualize; must match the ``time`` coordinate in ``zarr_stdized`` and ``zarr_ds``.

    Notes
    -----
    - This function expects the following globals to exist in the module scope:
        * ``datadir``: directory where ``{zarr_label}.npy`` lives,
        * ``zarr_ds``: the source dataset containing the observed variable
          ``'CHL_cmes-level3'`` and flag layers (``land_flag``, ``real_cloud_flag``,
          ``valid_CHL_flag``, ``fake_cloud_flag``).
    - Predictions are unstandardized with :func:`utils.unstdize`.
    - The hard-coded map extent corresponds to the Arab Sea region
      ``[lon_min, lon_max, lat_min, lat_max] = [42, 101.75, -11.75, 32]``.

    See Also
    --------
    utils.unstdize : Convert standardized values back to original (log-scale) units.
    utils.compute_mae : Mean absolute error (NaN-aware).
    utils.compute_mse : Mean squared error (NaN-aware).

    Example
    -------
    >>> # in your package code, with datadir and zarr_ds defined at module scope:
    >>> plot_prediction_observed(zarr_stdized, zarr_label="2015_3_ArabSea_full_2days",
    ...                          model=unet_model, date_to_predict="2015-03-15")
    """
    mean_std = np.load(f'{datadir}/{zarr_label}.npy', allow_pickle='TRUE').item()
    mean, std = mean_std['CHL'][0], mean_std['CHL'][1]

    zarr_date = zarr_stdized.sel(time=date_to_predict)

    # Build model input X from all standardized variables except "CHL"
    X = []
    X_vars = list(zarr_stdized.keys())
    X_vars.remove('CHL')
    for var in X_vars:
        var = zarr_date[var].to_numpy()
        X.append(np.where(np.isnan(var), 0.0, var))
    X = np.array(X)
    X = np.moveaxis(X, 0, -1)  # (C,H,W) -> (H,W,C)

    # Observed log(Chl-a)
    true_CHL = np.log(zarr_ds.sel(time=date_to_predict)['CHL_cmes-level3'].to_numpy())

    # Apply fake-cloud mask to observation for display
    fake_cloud_flag = zarr_date.fake_cloud_flag.to_numpy()
    masked_CHL = np.where(fake_cloud_flag == 1, np.nan, true_CHL)

    # Predict (standardized), then unstandardize to log-scale using utils.unstdize
    predicted_CHL = model.predict(X[np.newaxis, ...], verbose=0)[0]
    predicted_CHL = predicted_CHL[:, :, 0]
    predicted_CHL = unstdize(predicted_CHL, mean, std)

    # Keep NaN wherever observation is NaN (for fair visual diff)
    predicted_CHL = np.where(np.isnan(true_CHL), np.nan, predicted_CHL)
    diff = true_CHL - predicted_CHL

    # Flags panel data (0=land/real cloud, 1=fake cloud, 2=observed valid)
    flag = np.zeros(true_CHL.shape)
    flag = np.where(zarr_date['land_flag'] == 1, 0, flag)
    flag = np.where(zarr_date['valid_CHL_flag'] == 1, 2, flag)
    flag = np.where(zarr_date['real_cloud_flag'] == 1, 0, flag)
    flag = np.where(zarr_date['fake_cloud_flag'] == 1, 1, flag)

    # Color limits matched between observed and predicted
    vmax = np.nanmax((true_CHL, predicted_CHL))
    vmin = np.nanmin((true_CHL, predicted_CHL))

    extent = [42, 101.75, -11.75, 32]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    im0 = axes[0, 0].imshow(true_CHL, vmin=vmin, vmax=vmax, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(), interpolation='nearest')
    axes[0, 0].add_feature(cfeature.COASTLINE)
    axes[0, 0].set_xlabel('longitude'); axes[0, 0].set_ylabel('latitude')
    axes[0, 0].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[0, 0].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[0, 0].set_title('Observed Level-3 log Chl-a', size=14)

    im1 = axes[0, 1].imshow(flag, extent=extent, origin='upper',
                            transform=ccrs.PlateCarree())
    axes[0, 1].add_feature(cfeature.COASTLINE, color='white')
    axes[0, 1].set_xlabel('longitude'); axes[0, 1].set_ylabel('latitude')
    axes[0, 1].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[0, 1].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[0, 1].set_title('Land, Cloud, and Observed Flags After Applying Fake Cloud', size=13)

    im2 = axes[1, 0].imshow(predicted_CHL, vmin=vmin, vmax=vmax, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(), interpolation='nearest')
    axes[1, 0].add_feature(cfeature.COASTLINE, color='white')
    axes[1, 0].imshow(np.where(flag == 1, np.nan, flag), vmax=2, vmin=0,
                      extent=extent, origin='upper', interpolation='nearest', alpha=1)
    axes[1, 0].set_xlabel('longitude'); axes[1, 0].set_ylabel('latitude')
    axes[1, 0].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[1, 0].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    axes[1, 0].set_title('Predicted log Chl-a from U-Net', size=14)

    vmin2, vmax2 = -1, 1
    im3 = axes[1, 1].imshow(diff, vmin=vmin2, vmax=vmax2, extent=extent,
                            origin='upper', transform=ccrs.PlateCarree(),
                            cmap=plt.cm.RdBu, interpolation='nearest')
    axes[1, 1].add_feature(cfeature.COASTLINE)
    axes[1, 1].set_xlabel('longitude'); axes[1, 1].set_ylabel('latitude')
    axes[1, 1].set_xticks(np.arange(42, 102, 10), crs=ccrs.PlateCarree())
    axes[1, 1].set_yticks(np.arange(-12, 32, 5), crs=ccrs.PlateCarree())
    # (Optional) show quick metrics using utils helpers
    mae = compute_mae(true_CHL, predicted_CHL)
    mse = compute_mse(true_CHL, predicted_CHL)
    axes[1, 1].set_title(f'Difference (obs − pred)\nMAE={mae:.3f}, MSE={mse:.3f}', size=13)

    fig.subplots_adjust(right=0.76)
    cbar1_ax = fig.add_axes([0.79, 0.14, 0.025, 0.72]); fig.colorbar(im0, cax=cbar1_ax).ax.set_ylabel(
        'log Chl-a (mg/m-3)', rotation=270, size=14, labelpad=16)
    cbar2_ax = fig.add_axes([0.86, 0.14, 0.025, 0.72]); fig.colorbar(im1, cax=cbar2_ax).ax.set_ylabel(
        'land & real cloud = 0, fake cloud = 1, observed after masking = 2', rotation=270, size=14, labelpad=20)
    cbar3_ax = fig.add_axes([0.94, 0.14, 0.025, 0.72]); fig.colorbar(im3, cax=cbar3_ax).ax.set_ylabel(
        'difference in log Chl-a', rotation=270, size=14, labelpad=16)

    plt.show()
