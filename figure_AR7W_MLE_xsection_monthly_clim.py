# Rowan Brown, 3 Jun 2026
# Created on the suggestion of one reviewer; copied from the other
# AR7W cross section script (which creates a figure for the methods section)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as feature


def ANHA4_sections():
    """Creates example cross section figures showing isopycnals and the
    SMLE SF."""

    print("Beginning: Cross section calculations")

    # Init figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1', 'ax2', 'ax7', 'ax8'],
              ['ax3', 'ax4', 'ax9', 'a10'],
              ['ax5', 'ax6', 'a11', 'a12'],]
    fig, axd = plt.subplot_mosaic(layout, figsize=(19*cm, 12*cm))
    ax1, ax2, ax3, ax4 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4']
    ax5, ax6, ax7, ax8 = axd['ax5'], axd['ax6'], axd['ax7'], axd['ax8']
    ax9, a10, a11, a12 = axd['ax9'], axd['a10'], axd['a11'], axd['a12']

    def plot_xsection(run, month, ax, letter):

        fp = 'cross_section_'+run+'_'+'monthly_clim.nc'
        ds = xr.open_dataset(fp)
        ds = ds.sel(month=month)
        ds = ds.where(ds['vosaline'] != 0, drop=True)

        ds['pot_dens'] = ds['pot_dens']-1000  # Sigma notation

        cmap = mpl.colormaps.get_cmap('Blues')
        cmap.set_bad('cadetblue')
        cvmin, cvmax = 27, 27.775
        levels = np.linspace(cvmin, cvmax, 20)
        p2 = ds['pot_dens'].plot.contourf(
            ax=ax, add_colorbar=False, cmap='cividis_r', vmin=cvmin,
            vmax=cvmax, levels=levels, rasterized=True)
        p1 = ds['Psi'].plot.pcolormesh(
            ax=ax, add_colorbar=False, vmin=0,
            vmax=10, cmap=cmap, rasterized=True)
        ds['pot_dens'].plot.contour(
            ax=ax, add_colorbar=False, cmap='k', vmin=cvmin,
            vmax=cvmax, levels=levels, linewidths=0.5)
        ax.set_title('')
        ax.set_xticks([0, 240, 480, 720, 958])
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()
        ax.grid(visible=True, axis='both', color='grey', lw=1, alpha=0.75)

        month_dict = {12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar",
                      4: "Apr", 5: "May"}
        ax.set_title(month_dict[month], fontsize=12, pad=-10)

        run_dict = {"EPM155": "GDPS-TS", "EPM156": "ERAI-TS"}
        ax.text(
            0.025,
            0.045,
            run_dict[run],
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            va='bottom',
            ha='left',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.6,
                boxstyle='square,pad=0.1'
            )
        )

        ax.text(
            0.15,
            0.925,
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
    p, p2, ds, levels = plot_xsection("EPM155", 12, ax1, 'a')
    p, p2, ds, levels = plot_xsection("EPM155",  1, ax2, 'b')
    p, p2, ds, levels = plot_xsection("EPM155",  2, ax3, 'c')
    p, p2, ds, levels = plot_xsection("EPM155",  3, ax4, 'd')
    p, p2, ds, levels = plot_xsection("EPM155",  4, ax5, 'e')
    p, p2, ds, levels = plot_xsection("EPM155",  5, ax6, 'f')

    p, p2, ds, levels = plot_xsection("EPM156", 12, ax7, 'g')
    p, p2, ds, levels = plot_xsection("EPM156",  1, ax8, 'h')
    p, p2, ds, levels = plot_xsection("EPM156",  2, ax9, 'i')
    p, p2, ds, levels = plot_xsection("EPM156",  3, a10, 'j')
    p, p2, ds, levels = plot_xsection("EPM156",  4, a11, 'k')
    p, p2, ds, levels = plot_xsection("EPM156",  5, a12, 'l')

    # Adding correct x-axis ticks
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

    # Dealing with the axes and ticks
    for ax in [ax2, ax4, ax7, ax8, ax9, a10]:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_ticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax5.xaxis.set_ticklabels(xticks)
    ax5.set_xlabel('')
    ax5.set_ylabel('')
    for ax in [ax6, a11, a12]:
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels(xticks)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Labels
    fig.text(
        0.5,
        0.03,
        r'Distance along the AR7W section ($km$)',
        fontsize=12,
        ha='center')
    fig.text(
        0.04,
        0.35,
        r'Depth ($m$)',
        fontsize=12,
        ha='center',
        rotation=90)

    # Adding colourbar
    cbar_ax = fig.add_axes([0.2, 0.9, 0.6, 0.025])
    cb = fig.colorbar(
        p,
        cax=cbar_ax,
        orientation='horizontal',
        format='%.1f',
        extend='max')
    cb.ax.set_title(
        r"Cross-sections of the SMLEp streamfunction, $\Psi$",
        fontsize=12)

    plt.subplots_adjust(
        left=0.125,
        bottom=0.15,
        right=0.95,
        top=0.79,
        wspace=0.15,
        hspace=0.35)

    name = 'figure_AR7W_MLE_xsection_monthly_clim.svg'
    plt.savefig(name)

    print("Saved: "+name)


if __name__ == "__main__":
    ANHA4_sections()
