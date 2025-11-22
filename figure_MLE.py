# Looks at the MLE streamfunction
# Currently makes maps; doesn't seem useful at this stage to use time series
# Rowan Brown
# 14 Jul 2025

import pandas as pd
import xarray as xr
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import reduce
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import matplotlib.colors as colors


def mle_plot():
    """Create plot of MLE characteristics."""
    # Note that the reason we use 6 years of output is because that
    # is what we reran with the MLE diagnostics

    print("Beginning: MLE plot")

    # For mapping
    westLon = -63
    eastLon = -43.5
    northLat = 63
    southLat = 53

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1', 'ax1', 'ax1', '.', 'ax2', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax1', '.', 'ax2', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax1', '.', 'ax2', 'ax2', 'ax2'],
              ['ax3', 'ax3', 'ax3', 'ax3', '.', 'ax4', 'ax4'],
              ['ax3', 'ax3', 'ax3', 'ax3', '.', 'ax4', 'ax4']]
    proj = ccrs.AlbersEqualArea(
        central_longitude=-55,
        central_latitude=50,
        standard_parallels=(southLat, northLat))
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(19*cm, 15*cm),
        per_subplot_kw={("ax1", "ax2"): {"projection": proj}})
    ax1, ax2, ax3, ax4 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4']

    def open_processed_data(run, var):
        if var == 'mle':
            fp = 'MLE_psi_time_series_' + run + '.nc'
            df = xr.open_dataarray(fp).drop_vars(
                'time_centered').to_dataframe(run)
        elif var == 'mld':
            fp = 'ls3k_MLD_mean_' + run + '.nc'
            df = xr.open_dataarray(fp).drop_vars(
                ['deptht', 'time_centered']).to_dataframe(run)
        df = df.reset_index()
        df['time_counter'] = df['time_counter'].astype(str)
        df['time_counter'] = df['time_counter'].map(
            lambda date_string: datetime.strptime(
                date_string, '%Y-%m-%d %H:%M:%S'))
        df['time_counter'] = pd.to_datetime(df['time_counter'])
        return df

    # For the time series plot
    runs = ['EPM155', 'EPM156']
    Cs = plt.cm.viridis([0.8, 0.8])
    LSs = ['-', '--']
    htchs = ["", "///"]

    # For combining runs into one df (for the time series plot)
    def merge_dfs(dfs):
        merged = reduce(
            lambda left, right: pd.merge(
                left, right, on=['time_counter'], how='inner'), dfs)
        return merged

    # Actually now opening the MLD data for the time series plot
    df_mld = []
    for run in runs:
        df_temp = open_processed_data(run, 'mld')
        df_mld.append(df_temp)
    df_mld = merge_dfs(df_mld)
    df_mld = df_mld.set_index('time_counter')
    df_mld = df_mld.loc['2012-01-01':'2017-12-31']

    # And actually now opening the MLE data for the time series plot
    df = []
    for run in runs:
        df_temp = open_processed_data(run, 'mle')
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.set_index('time_counter')
    df = df.loc['2012-01-01':'2017-12-31']

    # Specifying that we're looking at a climatology
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months_reordered = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb',
                        'Mar', 'Apr', 'May', 'Jun', 'Jul']

    # Plotting the time series figure
    df_plot = df.groupby(df.index.month).mean()
    df_plot.index = months
    df_plot = df_plot.reindex(months_reordered)
    df_plot.plot(ax=ax3, color=Cs, style=LSs, rot=0, legend=False, lw=1.5)
    ax3.set_ylabel('$Ψ$ ($m^2$ $s^{-1}$)', fontdict={'fontsize': 12})
    ax3.set_title('Monthly mean $Ψ$ and MLD', fontdict={'fontsize': 12})
    ax3.set_xlabel(None)
    ax3.xaxis.set_tick_params(labelsize=9)
    ax3.yaxis.set_tick_params(labelsize=9)
    ax3.set_xticks(range(len(months_reordered)), months_reordered, rotation=0)
    ax3.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.text(
        0.075,
        0.95,
        'c',
        transform=ax3.transAxes,
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
    ax3_twin = ax3.twinx()
    df_mld_plot = df_mld.groupby(df_mld.index.month).mean()
    df_mld_plot.index = months
    df_mld_plot = df_mld_plot.reindex(months_reordered)
    df_mld_plot.plot(
        ax=ax3_twin,
        color='black',
        style=LSs,
        legend=False,
        lw=0.75
    )
    ax3_twin.set_ylabel('MLD ($m$)', fontdict={'fontsize': 12})

    # Dealing with legends
    ax3.legend(
        title='$Ψ$',
        labels=['GDPS-TM', 'ERAI-TM'],
        loc='upper left',
        shadow=True,
        bbox_to_anchor=(0.1, 0, 1., 1),
        fontsize=9
    )
    ax3_twin.legend(
        title='MLD',
        labels=['GDPS-TM', 'ERAI-TM'],
        shadow=True,
        loc='upper right',
        fontsize=9
    )

    # Adding a mean plot
    means = df.mean(axis=0)
    means.plot.bar(
        ax=ax4,
        xlabel='',
        color=Cs,
        hatch=htchs,
        width=1,
        edgecolor='w',
        label='',
        legend=False,
        rot=0,
        zorder=90
    )
    ax4.set_title('6-year mean $Ψ$', fontdict={'fontsize': 12})
    ax4.set_ylabel('$Ψ$ ($m^2$ $s^{-1}$)', fontdict={'fontsize': 12})
    ax4.set_xticks([0, 1], ['GDPS\n-TM', 'ERAI\n-TM'], rotation=0)
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax4.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax4.xaxis.set_tick_params(labelsize=9)
    ax4.yaxis.set_tick_params(labelsize=9)
    ax4.set_ylim(bottom=0, top=1)
    labls = [means['EPM155'], means['EPM156']]
    kwargs = {'rotation': 0}
    ax4.bar_label(
        ax4.containers[0],
        labels=[str(f'{i:.3f}')[:5] for i in labls],
        padding=3,
        **kwargs
    )
    ax4.text(
        0.175,
        0.95,
        'd',
        transform=ax4.transAxes,
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
    for spine in ax4.spines.values():
        spine.set_zorder(120)

    # Plotting maps (GDPS-TM)
    run = 'EPM155'
    path = 'MLE_psi_max_map_' + run + '.nc'
    uber_ds = xr.open_dataset(path)
    ds = uber_ds.isel(time_counter=5)
    land_50m = feature.NaturalEarthFeature(
        'physical', 'land', '50m', edgecolor='black', facecolor='gray')
    ax1.set_extent(
        [westLon, eastLon, southLat, northLat],
        crs=ccrs.PlateCarree()
    )
    ax1.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax1.coastlines(resolution='50m')
    gl = ax1.gridlines(draw_labels=True, dms=False, x_inline=False,
                       y_inline=False, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.ylocator = mticker.FixedLocator([50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-45, -55, -65])
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax1.pcolormesh(
        ds.nav_lon_grid_T,
        ds.nav_lat_grid_T,
        ds['Psi_max_map'],
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=100),
        rasterized=True
    )
    ax1.set_title(r'GDPS-TM: 2016 maximum $Ψ$', fontdict={'fontsize': 12})
    ax1.text(
        0.115,
        0.95,
        'a',
        transform=ax1.transAxes,
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

    # Copy+pasted from above :(
    run = 'EPM156'
    path = 'MLE_psi_max_map_' + run + '.nc'
    uber_ds = xr.open_dataset(path)
    ds = uber_ds.isel(time_counter=5)
    ax2.set_extent(
        [westLon, eastLon, southLat, northLat],
        crs=ccrs.PlateCarree())
    ax2.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax2.coastlines(resolution='50m')
    gl = ax2.gridlines(draw_labels=True, dms=False, x_inline=False,
                       y_inline=False, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.ylocator = mticker.FixedLocator([50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-45, -55, -65])
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    p = ax2.pcolormesh(
        ds.nav_lon_grid_T,
        ds.nav_lat_grid_T,
        ds['Psi_max_map'],
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=100),
        rasterized=True)
    ax2.set_title(r'ERAI-TM: 2016 maximum $Ψ$', fontdict={'fontsize': 12})
    ax2.text(
        0.115,
        0.95,
        'b',
        transform=ax2.transAxes,
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

    # Adding a colorbar for the maps
    axins = inset_axes(ax1,
                       width="7.5%",
                       height="100%",
                       loc="center right",
                       bbox_to_anchor=(0.27, 0, 1., 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,)
    fig.colorbar(p, cax=axins, orientation='vertical')
    axins.set_ylabel('$Ψ$ ($m^2$ $s^{-1}$)', size=12)
    axins.set_yticklabels(['0', '1', '10', '100'])
    axins.yaxis.set_label_position("left")

    plt.subplots_adjust(
        hspace=0.2,
        wspace=1,
        top=0.96,
        left=0.08,
        right=0.95,
        bottom=0.08
    )
    name = 'figure_MLE.svg'
    plt.savefig(name, dpi=600)

    print("Saved: "+name)


if __name__ == '__main__':
    mle_plot()
