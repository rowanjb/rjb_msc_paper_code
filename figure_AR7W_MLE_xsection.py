# Rowan Brown, 14 Aug 2025

import xarray as xr
import numpy as np
from metpy.interpolate import geodesic
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as feature


def ANHA4_sections():
    """Creates example cross section figures showing isopycnals and the
    MLE SF."""

    print("Beginning: Cross section calculations")

    # Init figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['.', '.', '.', 'ax5'],
              ['.', '.', '.', 'ax5'],
              ['.', '.', '.', 'ax5'],
              ['.', '.', '.', '.'],
              ['ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax2', 'ax2'],
              ['ax3', 'ax3', 'ax4', 'ax4'],
              ['ax3', 'ax3', 'ax4', 'ax4'],
              ['ax3', 'ax3', 'ax4', 'ax4'],
              ['ax3', 'ax3', 'ax4', 'ax4']]
    westLon, eastLon, northLat, southLat = -65, -40, 67, 51
    projection = ccrs.AlbersEqualArea(
        central_longitude=-55,
        central_latitude=50,
        standard_parallels=(southLat, northLat)
    )
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(19*cm, 15*cm),
        per_subplot_kw={("ax5"): {"projection": projection}}
    )
    ax1, ax2, ax3, ax4 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4']
    ax5 = axd['ax5']

    # Start and end coordinates of the AR7W cells
    vertices_lon = [-56.458036, -48.036965]
    vertices_lat = [53.410189, 60.733433]

    def plot_xsection(run, date, ax, letter):
        
        fp = 'cross_section_'+run+'_'+date+'.nc'
        ds = xr.open_dataset(fp)
        ds = ds.isel(time_counter=0)
        ds = ds.where(ds['vosaline'] != 0, drop=True)

        ds['pot_dens'] = ds['pot_dens']-1000  # Sigma notation

        cmap = mpl.colormaps.get_cmap('Greys')
        cmap.set_bad('cadetblue')
        cvmin, cvmax = 25, 27.775
        levels = np.linspace(cvmin, cvmax, 60)
        p2 = ds['pot_dens'].plot.contourf(
            ax=ax, add_colorbar=False, cmap='cividis_r', vmin=cvmin,
            vmax=cvmax, levels=levels, rasterized=True)
        p1 = ds['Psi'].plot.pcolormesh(
            ax=ax, add_colorbar=False, vmin=0,
            vmax=2.7, cmap=cmap, rasterized=True)
        ds['pot_dens'].plot.contour(
            ax=ax, add_colorbar=False, cmap='cividis_r', vmin=cvmin,
            vmax=cvmax, levels=levels, linewidths=0.2)
        ax.set_title('')
        ax.set_xticks([0, 240, 480, 720, 958])
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()
        ax.grid(visible=True, axis='both', color='grey', lw=0.5)

        run_dict = {"EPM155": "CGRF-\nTS", "EPM156": "ERAI-\nTS"}
        ax.text(
            0.025,
            0.025,
            run_dict[run],
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            va='bottom',
            ha='left'
        )

        ax.text(
            0.085,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='right',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='circle,pad=0.1'
            )
        )

        return p1, p2, ds, levels
    
    # Plotting
    letters = [['b', 'c',], ['d', 'e']]
    for row, run in enumerate(['EPM155', 'EPM156']):
        for col, date in enumerate(['y2013m05d15', 'y2013m07d04']):
            ax = [[ax1, ax2], [ax3, ax4]][row][col]
            letter = letters[row][col]
            p, p2, ds, levels = plot_xsection(run, date, ax, letter)
    
    # Adding dates
    ax1.text(
        0.5,
        1.1,
        '15 May 2013',
        transform=ax1.transAxes,
        fontsize=12,
        va='center',
        ha='center'
    )
    ax2.text(
        0.5,
        1.1,
        '4 July 2013',
        transform=ax2.transAxes,
        fontsize=12,
        va='center',
        ha='center'
    )

    # Dealing with the axes and ticks
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax2.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    old_xticks = [0, 240, 480, 720, 958]
    xticks = [
        int(round(d, -1)) 
        for d 
        in ds['dists'].sel(index=old_xticks).values/1000
    ]
    ds['dists'].isel(index=0).values
    xticks = [
        round(np.floor(d-ds['dists'].isel(index=0).values/1000)) 
        for d 
        in xticks
    ]
    ax3.xaxis.set_ticklabels(xticks)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax4.axes.yaxis.set_ticklabels([])
    ax4.axes.xaxis.set_ticklabels(xticks)
    ax4.set_ylabel('')
    ax4.set_xlabel('')
    fig.text(
        0.5,
        0.025,
        r'Distance along AR7W section ($km$)',
        fontsize=12,
        ha='center')
    fig.text(
        0.045,
        0.3,
        r'Depth ($m$)',
        fontsize=12,
        ha='center',
        rotation=90)

    # Adding colourbars
    cbar_ax = fig.add_axes([0.15, 0.87, 0.5, 0.025])
    cb = fig.colorbar(
        p,
        cax=cbar_ax,
        orientation='horizontal',
        format='%.1f')
    cb.ax.set_title(
        r"Cross-sections of the MLE streamfunction, $\Psi$",
        fontsize=12)
    cbar_ax = fig.add_axes([0.15, 0.73, 0.5, 0.025])
    cb = fig.colorbar(
        p2,
        cax=cbar_ax,
        orientation='horizontal',
        format='%.2f',
        extend='neither')
    cb.ax.set_title(r"Isopycnals ($kg$ $m^{-3}$)", fontsize=12)
    cb.ax.minorticks_off()
    cb.set_ticks(levels[::9])

    # Adding map
    land_50m = feature.NaturalEarthFeature(
        'physical',
        'land',
        '50m',
        edgecolor='black',
        facecolor='cadetblue'
    )
    ax5.set_extent(
        [westLon, eastLon, southLat, northLat],
        crs=ccrs.PlateCarree()
    )
    ax5.set_title("AR7W section", fontsize=12, pad=-10)
    ax5.add_feature(land_50m, color="cadetblue")
    ax5.coastlines(resolution='50m')
    ax5.plot(
        vertices_lon,
        vertices_lat,
        transform=ccrs.PlateCarree(),
        color='black'
    )
    gl = ax5.gridlines(
        draw_labels=True,
        dms=False,
        x_inline=False,
        y_inline=False,
        linewidth=0.5
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.ylocator = mticker.FixedLocator([50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-45, -55, -65])
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax5.text(
        0.3,
        0.9,
        'a',
        transform=ax5.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='right',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='circle,pad=0.1'
        )
    )

    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0.15,
        hspace=0.75)

    name = 'figure_AR7W_MLE_xsection.svg'
    plt.savefig(name)

    print("Saved: "+name)


if __name__ == "__main__":
    ANHA4_sections()
