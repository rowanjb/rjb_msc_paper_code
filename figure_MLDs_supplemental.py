# Rowan Brown 13 Aug 2025

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import cartopy.crs as ccrs
import cartopy.feature as feature
from cftime import DatetimeNoLeap


def plot_MLDs_supplemental_figure():
    """Plot of domain(ish)-scale MLDs for the appendix of the paper.
    Include Argo but not LAB60, since the domain is too small.
    Uses the datasets created with stratification.py"""

    print("Plotting MLD supplemental figure")

    # Define figure and calculation limits
    nlat, slat = 70, 10
    wlon, elon = -90, 0

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1', 'ax2'],
              ['ax3', 'ax4'],
              ['ax5', 'ax6']]
    proj = ccrs.PlateCarree(central_longitude=-35)
    fig, axd = plt.subplot_mosaic(layout, subplot_kw={'projection': proj})
    fig.set_figwidth(19*cm)
    fig.set_figheight(19*cm)
    ax1, ax2, ax3 = axd['ax1'], axd['ax2'], axd['ax3']
    ax4, ax5, ax6 = axd['ax4'], axd['ax5'], axd['ax6']

    '''
    def calculate_area_weighted_mean_and_sd(da):
        print("Calculating mean and stddev")
        mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc')
        areas = mesh['e1t']*mesh['e2t']
        weights = areas/areas.mean()
        weights = weights.fillna(0).isel(t=0).rename(
            {'y': 'y_grid_T', 'x': 'x_grid_T'})#.where(da.notnull())
        m = da.weighted(weights).mean(skipna=True).values
        return m
    '''

    # Function for plotting MLD anomalies
    def plot_anom_map(da, argo, ax, run, letter, weights):

        # First cut down the time (not sure if necessary, but why not?)
        da = da.where(
            (da['time'] > DatetimeNoLeap(2008, 1, 1, 0, 0, 0)) &
            (da['time'] < DatetimeNoLeap(2018, 1, 1, 0, 0, 0)))
        da = da.where(argo)  # Mask only where we have Argo
        da = da - argo  # Calculate the differences

        mean_diff = da.weighted(weights).mean(skipna=True).values
        stddev = da.weighted(weights).std(skipna=True).values

        land_50m = feature.NaturalEarthFeature(
            'physical', 'land', '50m', edgecolor='black', facecolor='gray')
        ax.set_extent([wlon, elon, slat, nlat], crs=ccrs.PlateCarree())
        ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
        ax.coastlines(resolution='50m')
        gl = ax.gridlines(
            draw_labels=True,
            dms=False,
            x_inline=False,
            y_inline=False,
            linewidth=1,
            alpha=0.75
        )
        gl_label_dict = {  # [top, bottom, left, right]
            ax1: [False, False, True, False],
            ax2: [False, False, False, True],
            ax3: [False, False, True, False],
            ax4: [False, False, False, True],
            ax5: [False, True, True, False],
            ax6: [False, True, False, True]
        }
        gl.top_labels = gl_label_dict[ax][0]
        gl.bottom_labels = gl_label_dict[ax][1]
        gl.left_labels = gl_label_dict[ax][2]
        gl.right_labels = gl_label_dict[ax][3]
        gl.rotate_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

        p = ax.pcolormesh(
            argo['lons'],
            argo['lats'],
            da.mean('time', skipna=True),
            transform=ccrs.PlateCarree(),
            cmap='BrBG',
            rasterized=True,
            norm=c.SymLogNorm(
                linthresh=10,
                linscale=1,
                vmin=-1000,
                vmax=1000
            )
        )
        ax.text(
            0.05,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='circle,pad=0.1'
            )
        )
        run_dict = {
            'EPM157': 'GDPS-C',
            'EPM158': 'ERAI-C',
            'EPM151': 'GDPS-T',
            'EPM152': 'ERAI-T',
            'EPM155': 'GDPS-TS',
            'EPM156': 'ERAI-TS'
        }
        label = (
            run_dict[run] + '\n'
            + f'Mean: {float(mean_diff):.3g} m\n'
            + f'Std. dev.: {float(stddev):.3g} m'
        )
        ax.text(
            0.05,
            0.07,
            label,
            transform=ax.transAxes,
            fontsize=9,
            va='bottom',
            ha='left',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                alpha=1
            )
        )
        return p

    # Argo MLD data
    # Note to self: mean MLDs from...
    # - the 1 degree 2018 climatology for the NA: 58.30867742030664;
    # - my 1/4 degree dataset for the same region but not including
    #   before 2008: 60.56136529125293, i.e., pretty dang close :)
    fp = 'Argo_mld_NorthAtlantic.nc'
    argo_ds = xr.open_dataset(fp)
    argo_ds = argo_ds.rename({
        "__xarray_dataarray_variable__": "Argo",
        'y': 'y_grid_T',
        'x': 'x_grid_T'})
    argo_gridded = argo_ds.set_index(da_mld=["time", "y_grid_T", "x_grid_T"]).unstack("da_mld")
    argo_gridded = argo_gridded.where(
        (argo_gridded['time'] > DatetimeNoLeap(2008, 1, 1, 0, 0, 0)) &
        (argo_gridded['time'] < DatetimeNoLeap(2018, 1, 1, 0, 0, 0)))
    argo_da = argo_gridded['Argo'].mean('time', skipna=True)

    '''
    proj = ccrs.PlateCarree(central_longitude=-35)
    fig, ax = plt.subplots(subplot_kw={'projection': proj})

    land_50m = feature.NaturalEarthFeature(
        'physical', 'land', '50m', edgecolor='black', facecolor='gray')
    ax.set_extent([wlon, elon, slat, nlat], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.coastlines(resolution='50m')
    p = ax.pcolormesh(
        mesh_lon,
        mesh_lat,
        da_158-da,
        transform=ccrs.PlateCarree(),
        cmap='BrBG',
        rasterized=True,
        norm=c.SymLogNorm(
            linthresh=10,
            linscale=1,
            vmin=-1000,
            vmax=1000
        )
    )

    argo = ds_argo['yearly_mean'].mean(dim='year')
    std_dev = ds_argo['std_dev']
    '''

    # We also need to open the mesh file to get the lats and lons
    # We can also calculate the weights for area-averages now
    mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc')
    mesh = mesh.rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    areas = mesh['e1t']*mesh['e2t']
    weights = (areas/areas.mean()).fillna(0).isel(t=0)
    lats = mesh.nav_lat.sel(
        y_grid_T=argo_gridded.y_grid_T, x_grid_T=argo_gridded.x_grid_T)
    lons = mesh.nav_lon.sel(
        y_grid_T=argo_gridded.y_grid_T, x_grid_T=argo_gridded.x_grid_T)
    weights = mesh.nav_lon.sel(
        y_grid_T=argo_gridded.y_grid_T, x_grid_T=argo_gridded.x_grid_T)

    # Now we can actually add the lats and lons to the Argo dataarray
    # and then chop it down to only the North Atlantic
    argo_da = argo_da.assign_coords(lats=lats, lons=lons)
    argo_da = argo_da.where(
        (argo_da.lats < nlat) &
        (argo_da.lats > slat) &
        (argo_da.lons > wlon) &
        (argo_da.lons < elon)
    )

    # Plotting ANHA4 MLD data
    runs = ['EPM157', 'EPM158', 'EPM151', 'EPM152', 'EPM155', 'EPM156']
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for n, run in enumerate(runs):
        print("Plotting: "+run)
        ax = axes[n]
        letter = letters[n]
        fp = 'Argo_mld_NorthAtlantic_'+run+'.nc'
        ds = xr.open_dataset(fp)
        ds = ds.drop_vars([
            'nav_lat_grid_T', 'nav_lon_grid_T', 'time_centered',
            'time_counter', 'nav_lat', 'nav_lon'])
        ds = ds.rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
        ds_gridded = ds.set_index(da_mld=["time", "y_grid_T", "x_grid_T"]).unstack("da_mld")
        p = plot_anom_map(ds_gridded['somxlts'], argo_da, ax, run, letter, weights)

    # Colourbar
    cbar_ax = fig.add_axes([0.2, 0.91, 0.6, 0.025])
    cb = fig.colorbar(p, cax=cbar_ax, orientation='horizontal', format='%.0f')
    cb.ax.set_title("Anomaly of 10-yr mean MLDs ($m$)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    # Adjust bounding whitespace
    fig.subplots_adjust(
        bottom=0.09,
        top=0.85,
    )

    name = 'figure_MLDs_supplemental.svg'
    plt.subplots_adjust(hspace=0.04)
    plt.subplots_adjust(wspace=0.001)
    plt.savefig('test.png', dpi=600)
    plt.savefig(name, dpi=600)
    plt.close(fig)

    print("Saved :" + name)


if __name__ == "__main__":
    plot_MLDs_supplemental_figure()
