# Rowan Brown 13 Aug 2025

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import cartopy.crs as ccrs
import cartopy.feature as feature


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

    def calculate_area_weighted_mean(da):
        mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc')
        areas = mesh['e1t']*mesh['e2t']
        weights = areas/areas.mean()
        weights = weights.fillna(0)
        return da.weighted(weights).mean(skipna=True).values

    # Function for plotting MLD anomalies
    def plot_anom_map(da, argo, ax, run, letter):
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
        da = da.where(argo)  # Mask the model data only where we have Argo
        da = da - argo  # Calculate the differences
        mean_diff = calculate_area_weighted_mean(da)
        p = ax.pcolormesh(
            da.nav_lon_grid_T,
            da.nav_lat_grid_T,
            da,
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
        label = run_dict[run]+'\nMean: '+str(round(float(mean_diff), 2))+' m'
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
    fp = 'MLD_yearly_maps_NorthAtlantic_domain_Argo.nc'
    ds_argo = xr.open_dataset(fp).rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    argo = ds_argo['yearly_mean'].mean(dim='year')
    std_dev = ds_argo['std_dev']

    # == Revision: Calc the standard dev == #
    std_dev = std_dev.where(
        (std_dev.nav_lat_grid_T < nlat) &
        (std_dev.nav_lat_grid_T > slat) &
        (std_dev.nav_lon_grid_T > wlon) &
        (std_dev.nav_lon_grid_T < elon)
    )
    mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc')
    mesh = mesh.isel(t=0).rename({'y': 'y_grid_T', 'x': 'x_grid_T'})
    areas = mesh['e1t']*mesh['e2t']
    areas = areas.where(std_dev.sum('year') > 0)  # cheeky mask
    nbins = xr.where(std_dev > 0, 1, 0).sum('year')
    std_dev = ((std_dev**2).sum('year')/nbins**2)**0.5
    stddev_final = (std_dev**2)*(areas**2)
    area = areas.sum(dim=['y_grid_T', 'x_grid_T'])
    stddev_final = (stddev_final.sum(dim=['y_grid_T', 'x_grid_T'])/area**2)**0.5
    print("Std dev of the Argo mean: " + str(stddev_final.to_numpy()) + ' dbar')
    quit()

    # Masking the argo data because we don't want too close to the boundaries
    argo = argo.where(
        (argo.nav_lat_grid_T < nlat) &
        (argo.nav_lat_grid_T > slat) &
        (argo.nav_lon_grid_T > wlon) &
        (argo.nav_lon_grid_T < elon)
    )

    # Plotting ANHA4 MLD data
    runs = ['EPM157', 'EPM158', 'EPM151', 'EPM152', 'EPM155', 'EPM156']
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for n, run in enumerate(runs):
        print("Plotting: "+run)
        ax = axes[n]
        letter = letters[n]
        fp = 'MLD_yearly_maps_full_domain_'+run+'.nc'
        ds = xr.open_dataset(fp)
        ds = ds.sel(year=slice(2008, 2017))
        da = ds['yearly_mean'].mean(dim='year')
        p = plot_anom_map(da, argo, ax, run, letter)

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
    plt.savefig(name, dpi=600)
    plt.close(fig)

    print("Saved :" + name)


if __name__ == "__main__":
    plot_MLDs_supplemental_figure()
