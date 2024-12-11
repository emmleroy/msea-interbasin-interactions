"""
Plotting functions that take axes objects as inputs
"""

from datetime import datetime

import pandas as pd
import numpy as np
import cmaps
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely import geometry
from scipy.stats import linregress 

from src import utils, precip, models
from src.inputs import *

ensemble_members = models.CESM2_ensemble_members # List of CESM2 Ensemble Members

# Figure 1a
def plot_msea_precipation_climatology(prect_data_source,
                                      fig, ax,
                                      season="MAM",
                                      cmap=cmaps.CBR_wet,
                                      levels=[0,25,50,75,100,125,150,175,200],
                                      cbar_ticks=[0, 50, 100, 150, 200]):
    """Plot climatology of observed precipitation in Mainland Southeast Asia"""
    
    # Load and process input precipitation data
    file = prect_gridded_rain_gauge_source_to_file[prect_data_source]  
    
    ds = utils.open_dataset(file)
    da = ds["precip"].sel(time=slice("1951-01", "2015-12"))

    msea_precipitatio_map = precip.get_SEAM_map(da, monthly=True, detrend=False, anomaly=False)
    msea_precipitation_map_seasonal = msea_precipitatio_map.sel(time=msea_precipitatio_map.time.dt.season==season)
    msea_precipitation_map_seasonal = msea_precipitation_map_seasonal.resample(time="1Y").mean()

    # Draw filled contour plot
    im = ax.contourf(
            msea_precipitation_map_seasonal.lon,
            msea_precipitation_map_seasonal.lat,
            msea_precipitation_map_seasonal.mean(dim='time'),
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extend="max",
            levels=levels
        )

    # Coastlines, gridlines, and extent
    ax.coastlines(linewidth=0.5)
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color="gray", alpha=0.5
    )
    ax.set_extent([80,120,0,35], crs=ccrs.PlateCarree())

    # Xticks and Yticks
    ax.set_xticks([90, 110],crs=ccrs.PlateCarree())
    ax.set_xticklabels([90, 110],fontsize=8)
    ax.set_yticks([10,25],crs=ccrs.PlateCarree())
    ax.set_yticklabels([10,25],fontsize=8)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
  
    # Colorbar
    cax,kw = mpl.colorbar.make_axes(ax, location='right', pad=0.1, shrink=0.99, aspect=15)
    cbar = fig.colorbar(im, ticks=cbar_ticks, cax=cax, **kw)
    cbar.ax.set_title('mm/month', fontsize=8)
    cbar.outline.set_linewidth(0.5)

    # Draw box around MSEA region
    geom = geometry.box(minx=90,maxx=110, miny=10,maxy=25)
    ax.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=.5)

    return im


# Figure 1b
def plot_runcorr_statistics_timeseries(obs_nino34_sst_anomalies,
                                       obs_msea_prect_anomalies, ax,
                                       window=13):
    """Plot mean and range of observed running correlation between 
    ENSO (DJF SST anomalies) and MSEA (MAM precip. anomalies)"""

    # Calculate statistics of running correlation timeseries
    runcorr_mean, runcorr_max, runcorr_min = (
        utils.calculate_runcorr_statistics_timeseries(
        obs_nino34_sst_anomalies, obs_msea_prect_anomalies, window=window
        ))

    # Generate dates from '1951-12-31' to '2015-12-31' with an annual frequency
    dates0 = np.arange('1951-12-31', '2016-01-01', dtype='datetime64[Y]').astype('datetime64[D]')
    # Convert to Python datetime objects
    dates = dates0.astype('datetime64[D]').astype(object)

    ax.plot(dates,
            runcorr_mean,
            linestyle='-',
            color='red',
            linewidth=1,
            label="Mean"
        )

    ax.fill_between(
            dates,
            runcorr_min,
            runcorr_max,
            color='grey',
            alpha=0.5,
            label="Range"
        )

    ax.legend(fontsize=6, frameon=False, loc=[0.75,0.6])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Draw 90% confidence level for correlation coefficient of window length
    crit = utils.critical_r_value(window, confidence_level=0.90) # critical r-value at 90% confidence level
    ax.axhline(-crit, color="grey", linestyle='--', label=None, linewidth=0.5)
    ax.axhline(+crit, color="grey", linestyle='--', label=None, linewidth=0.5)

    # Shade in grey negative periods of the IPO
    ax.axvspan('1948', '1977', alpha=0.2, facecolor='grey', edgecolor=None)
    ax.axvspan('1999', '2023', alpha=0.2, facecolor='grey', edgecolor=None)

    # Add "+IPO" in the middle of each grey box at the top
    grey_midpoints = [datetime(1966, 1, 1), datetime(2011, 1, 1)]
    for midpoint in grey_midpoints:
        ax.text(midpoint, 1, '-IPO', ha='center', va='top',
        fontsize=6, transform=ax.get_xaxis_transform())

    # Add "-IPO" in the middle of each white space at the top
    white_midpoints = [datetime(1988, 1, 1)]
    for midpoint in white_midpoints:
        ax.text(midpoint, 1, '+IPO', ha='center', va='top',
        fontsize=6, transform=ax.get_xaxis_transform())

    # Draw horizontal line at 0
    ax.axhline(0, color="grey", linewidth=1, label=None)

    # Set x- and y- labels and axes
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(base=10))
    ax.set_xlim(pd.Timestamp("1950-01-01"), pd.Timestamp("2016-12-01"))
    ax.set_ylim([-1,0.75])
    ax.set_ylabel(None)
    ax.tick_params(axis='both', which='major', labelsize=8)
    return ax


# Figure 1c and 1d
def plot_regression_map(ax, regression_mean, 
                        sign_mask=None,
                        cmap=cmaps.NCV_blu_red,
                        levels=[-15, -12.5, -10, -7.5, -5, -2.5, -0.5, 0.5, 2.5, 5, 7.5, 10, 12.5, 15]):
    """Plot map of mean regression coefficients and (optionally) add
    hatching where the ensemble members / data pairs agree 
    on the sign of the regression"""

    # Draw filled contour plot
    im = ax.contourf(
                    regression_mean.lon,
                    regression_mean.lat,
                    regression_mean,
                    cmap=cmap,
                    norm=mpl.colors.CenteredNorm(),
                    transform=ccrs.PlateCarree(),
                    extend="both",
                    levels=levels,
                )

    # Add hatching where the data agree on the sign of the regression
    hatching = ax.pcolor(regression_mean.lon, 
                         regression_mean.lat,
                         regression_mean.where(sign_mask), 
                         hatch='/////', alpha=0, transform=ccrs.PlateCarree())
    hatching.set_linewidth(0.5)  # Adjust linewidth for hatches

    # Coastlines, gridlines, and extent
    ax.coastlines(linewidth=0.5)
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color="gray", alpha=0.5
    )
    ax.set_global()

    lon_formatter = LongitudeFormatter(zero_direction_label=True)

    ax.set_yticks([-60, 0, 60], crs=ccrs.PlateCarree())
    ax.set_xticks([60, 180, 300], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(lon_formatter)

    return im


# Figure 2a
def plot_runcorr_cesm_timeseries(correlations_da, 
                                 ax, 
                                 future_color="palevioletred",
                                 forcing_label="SSP3-7.0 (2015-2100)"):
    """Plot timeseries of running correlations from CESM2"""

    for ens in range(len(correlations_da.ensemble)):
        x = correlations_da['corr'].isel(ensemble=ens)
        ax.plot(
            x.time[:116],
            x.values[:116],
            linestyle='-',
            color='grey',
            alpha=0.4,
            linewidth=0.5,
        )
        ax.plot(
            x.time[115:201],
            x.values[115:201],
            linestyle='-',
            color=future_color,
            alpha=0.4,
            linewidth=0.5,
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ensemble_mean = correlations_da['corr'].mean(dim='ensemble')
    ax.plot(
            ensemble_mean.time,
            ensemble_mean,
            label="Ensemble Mean",
            linestyle='-',
            color='k',
            alpha=1,
            linewidth=1
        )

    line1 = Line2D([0], [0], color='grey', linewidth=0.5, linestyle='-')
    line2 = Line2D([0], [0], color=future_color, linewidth=0.5, linestyle='-')
    line3 = Line2D([0], [0], color='k', linewidth=1, linestyle='-')

    plt.gca().add_patch(line1)
    plt.gca().add_patch(line2)
    plt.gca().add_patch(line3)

    ax.legend([line1, line2, line3], 
              ['Historical (1900-2014)', f'{forcing_label}', 'Ensemble Mean'], 
              loc='upper left', frameon=False, fontsize=6)

    line1.set_visible(False)
    line2.set_visible(False)
    line3.set_visible(False)

    # Set x- and y- labels and axes
    ax.xaxis.set_minor_locator(mpl.dates.YearLocator(base=10))
    ax.set_xlim(-37000,37000)
    ax.set_ylim(-1,1)
    ax.set_ylabel("13-year run-corr.", fontsize=8)
    ax.set_xlabel("Year", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)  # Set tick params for ax1

    return ax


# Figure 2b
def draw_pdf(data, ax, xlabel, ylabel='Probability Density'):

    # Create the histogram with the 'viridis' colormap
    _, bins, patches = ax.hist(data, 
    bins=35, linewidth=0.5, edgecolor='black', alpha=0.7, density=True
    )

    # Color each bar with the 'viridis' colormap
    for i, patch in enumerate(patches):
        plt.setp(patch, 'facecolor', 'grey')

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Iterate over the patches to color the bottom and top quartiles
    for i, patch in enumerate(patches):
        # Check if the current bin falls into the bottom or top quartile
        if bins[i] < q1:
            patch.set_facecolor('dodgerblue')
        if  bins[i+1] > q3:
            patch.set_facecolor('coral')

    legend_handles = [
        Patch(facecolor='dodgerblue', alpha=0.7, label='Lower Quartile \n (Correlated)'),
        Patch(facecolor='coral', alpha=0.7, label='Upper Quartile \n (Uncorrelated)')
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, fontsize=6, frameon=False, loc='best')

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    ax.yaxis.set_ticks_position('right')  # Set ticks to the right side
    ax.yaxis.set_label_position('right')  # Set the y-label to the right side
    ax.spines['top'].set_position(('axes', 1.0))  # Move right spine to right
    ax.spines['left'].set_color('none')  # Hide the left spine

    return ax


# Figure 2c and 2d
def draw_quartile_anomalies(data, fig, ax, p_val, 
                            vmin, vmax, levels, 
                            ticks, cmap, cbar=False, 
                            label=None, MSEA_box=False):
    mpl.rcParams['hatch.linewidth'] = 0.4  # previous pdf hatch linewidth

    im = ax.contourf(
                    data.lon,
                    data.lat,
                    data,
                    cmap=cmap,
                    norm=mpl.colors.CenteredNorm(),
                    transform=ccrs.PlateCarree(),
                    extend="both",
                    levels=levels
                    )

    sig = ax.contourf(
                    data.lon,
                    data.lat,
                    data.where(p_val<0.001),
                    cmap=cmap,
                    norm=mpl.colors.CenteredNorm(),
                    transform=ccrs.PlateCarree(),
                    extend="both",
                    hatches=['/////'],
                    levels=levels
                    )

    ax.set_global()

    if MSEA_box is True:
        geom = geometry.box(minx=90, maxx=110, miny=10, maxy=25)
        ax.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='red', facecolor='none', linewidth=0.5)

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.coastlines(linewidth=0.5)
    ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color="gray", alpha=0.5
        )

    if cbar is True:
        cax,kw = mpl.colorbar.make_axes(ax, ticks=ticks, location='bottom', pad=0.05, shrink=0.80, aspect=25)
        cbar = fig.colorbar(im, cax=cax, label=label, **kw)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

    ax.set_xticks([60, 120, 180, 240, 300],
                                    crs=ccrs.PlateCarree())
    ax.set_xticklabels([60, 120, 180, 240, 300],fontsize=8)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_yticks([-60, -30, 0, 30, 60], 
                                crs=ccrs.PlateCarree())

    return im


# Helper Functions for Figure 3
def add_quiver(ax, diff_u, diff_v):
    skip = (slice(None, None, 8), slice(None, None, 8))
    lon2d, lat2d = np.meshgrid(diff_u.lon, diff_u.lat)
    q = ax.quiver(
        lon2d[skip], lat2d[skip],
        diff_u.values[skip], diff_v.values[skip],
        transform=ccrs.PlateCarree(),
        scale=8, scale_units='inches',
        color='black', headwidth=4, headlength=3,
        headaxislength=2, width=0.006
    )
    return q


def plot_sst_wind(ax, diff_sst, diff_u, diff_v):
    sst_levels = np.linspace(-0.5, 0.5, 21)
    im = ax.contourf(
        diff_sst.lon, diff_sst.lat, diff_sst,
        levels=sst_levels, cmap=cmaps.NCV_blu_red,
        norm=mpl.colors.CenteredNorm(), transform=ccrs.PlateCarree(),
        extend="both"
    )
    q = add_quiver(ax, diff_u, diff_v)
    return im, q


def add_rectangles(ax):
    regions = [
        {'minx': 50, 'maxx': 70, 'miny': -10, 'maxy': 10},    # Tropial West Indian Ocean
        {'minx': 100, 'maxx': 125, 'miny': -20, 'maxy': 20},  # Central region
        {'minx': 160, 'maxx': 210, 'miny': -5, 'maxy': 5}     # Western Pacific
    ]
    for region in regions:
        geom = geometry.box(**region)
        ax.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.5)


def plot_omega(ax, diff_omega):
    ax.invert_yaxis()
    diff_omega = diff_omega.sel(lon=slice(30, 240), lev=slice(100, 1000))
    om = ax.contourf(
        diff_omega.lon, diff_omega.lev, diff_omega * 1e3,
        cmap=cmaps.BlueWhiteOrangeRed_r, norm=mpl.colors.CenteredNorm(),
        extend="both", levels=np.arange(-6, 7, 1)
    )
    return om


def plot_prect_tmq(ax, diff_prect, diff_tmq):
    pc_levels = np.linspace(-10, 10, 11)
    pc = ax.contourf(
        diff_prect.lon, diff_prect.lat, diff_prect,
        levels=pc_levels, cmap='BrBG', extend='both',
        transform=ccrs.PlateCarree()
    )
    mc_levels = [-0.5, -0.25, 0, 0.25, 0.5]
    mc = ax.contour(
        diff_tmq.lon, diff_tmq.lat, diff_tmq,
        levels=mc_levels, colors='limegreen',
        linewidths=0.5, transform=ccrs.PlateCarree()
    )
    ax.clabel(mc, inline=True, fontsize=6, fmt='%1.1f', colors='limegreen')
    return pc


## Helper Functions for Figure 4 ##
def plot_linregress(x, y, ax, color, linewidth=1):
    """plot line of linear least-squares regression"""

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Generate x values for the trend line
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = intercept + slope * x_fit

    ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=linewidth, zorder=10)
    
    return slope


def annotate_slope(ax, x_data, y_data, x_pos, label_color, label=True):
    """Plot linear regression (linear least squared) and annotate slope."""
    slope = plot_linregress(
        x_data[~np.isnan(x_data)].flatten(), 
        y_data[~np.isnan(y_data)].flatten(), 
        ax=ax, color='black'
        )
    if label==True:
        ax.text(x_pos, 125, f'{slope:.0f} mm/month/°C', color=label_color, fontsize=6)

def select_and_shift_sst(sst_data, time_shift=1, start='1951', end='2015'):
    """Shift and select SST data within a specified time range.
    Shift SST by 1 so that we properly compare D(-1)JF(0) SSTs to MAM(0) prect."""
    sst_shifted = sst_data.shift(time=time_shift).sel(time=slice(start, end))
    return sst_shifted

def filter_enso_events(precip_anm, sst_shifted):
    """Separate El Niño and La Niña events based on SST conditions."""
    
    # El Niño when SST anm > 0
    elnino_sst = sst_shifted.where(sst_shifted > 0, drop=True).values
    elnino_prect = precip_anm.where(sst_shifted > 0, drop=True).values

    # La Niña when SST anm < 0
    lanina_sst = sst_shifted.where(sst_shifted < 0, drop=True).values
    lanina_prect = precip_anm.where(sst_shifted < 0, drop=True).values
    
    return elnino_sst, lanina_sst, elnino_prect, lanina_prect

def plot_enso_asymmetry_scatter(ax, elnino_sst, elnino_pre, lanina_sst, lanina_pre, el_color='red', la_color='blue'):
    """Plot scatter points for El Niño and La Niña events."""
    ax.scatter(elnino_sst, elnino_pre, marker='o', edgecolors=el_color, facecolors=el_color, s=6, linewidths=0.25, alpha=1, zorder=5)
    ax.scatter(lanina_sst, lanina_pre, marker='o', edgecolors=la_color, facecolors=la_color, s=6, linewidths=0.25, alpha=1, zorder=5)

# Figure 5
def plot_sst_trends_spatial(slope, ax, num_years, levels):
    slope = utils.mask_land(slope)

    trend_plot = ax.contourf(
        slope['lon'], slope['lat'], slope*num_years,
        cmap=cmaps.NCV_blu_red,
        levels=levels,
        transform=ccrs.PlateCarree(),
        norm=mpl.colors.CenteredNorm(),
        extend='neither'
    )
    return trend_plot