# Rowan Brown, 13 Aug 2025

import pandas as pd
import xarray as xr
from datetime import datetime as dt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import reduce
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker


def plot_MLDs_body_figure():
    """Plot of MLDs for the main body of the paper."""

    print("Beginning to plot the MLD figure (body)")

    # For mapping
    westLon = -63
    eastLon = -43.5
    northLat = 63
    southLat = 53

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1', 'ax1', 'ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax2', 'ax2'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax3', 'ax3'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax3', 'ax3'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax3', 'ax3'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax3', 'ax3'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax4', 'ax4'],
              ['.', '.', '.', '.', 'ax4', 'ax4'],
              ['.', '.', '.', '.', 'ax4', 'ax4'],
              ['.', '.', '.', '.', 'ax4', 'ax4'],
              ['ax5', 'ax5', 'ax6', 'ax6', 'ax7', 'ax7'],
              ['ax5', 'ax5', 'ax6', 'ax6', 'ax7', 'ax7'],
              ['ax5', 'ax5', 'ax6', 'ax6', 'ax7', 'ax7'],
              ['ax5', 'ax5', 'ax6', 'ax6', 'ax7', 'ax7']]
    proj = ccrs.AlbersEqualArea(
        central_longitude=-55,
        central_latitude=50,
        standard_parallels=(southLat, northLat)
    )
    fig, axd = plt.subplot_mosaic(
        layout,
        per_subplot_kw={
            ("ax2", "ax3", "ax4", "ax5", "ax6", "ax7"): {"projection": proj}
        }
    )
    fig.set_figwidth(19*cm)
    fig.set_figheight(19*cm)
    ax1 = axd['ax1']
    ax2 = axd['ax2']
    ax3 = axd['ax3']
    ax4 = axd['ax4']
    ax5 = axd['ax5']
    ax6 = axd['ax6']
    ax7 = axd['ax7']

    # Function for opening and processing the MLD data and storing it a pd df
    def open_processed_data(run):
        fp = 'ls3k_MLD_mean_'+run+'.nc'
        if 'LAB60' in run:
            df = xr.open_dataarray(fp).to_dataframe(run)
            df = df.reset_index()  # Turning the index into timestamp
            df['time_counter'] = df['time_counter'].astype(str)
            df['time_counter'] = df['time_counter'].map(
                lambda date_string: dt.strptime(date_string, '%Y-%m-%d')
            )
        elif 'EPM' in run:  # Note our NEMO4 runs are called EPM###
            ds = xr.open_dataarray(fp)
            df = ds.drop_vars(['deptht', 'time_centered']).to_dataframe(run)
            df = df.reset_index()
            df['time_counter'] = df['time_counter'].astype(str)
            df['time_counter'] = df['time_counter'].map(
                lambda date_string: dt.strptime(
                    date_string, '%Y-%m-%d %H:%M:%S'
                )
            )
        elif 'Argo' in run:  # Argo data
            df = xr.open_dataarray(fp).drop_vars(['deptht']).to_dataframe(run)
            df = df.reset_index()
            df['date'] = df['date'].astype(str)
            df['date'] = df['date'].map(
                lambda date_string: dt.strptime(date_string, '%Y-%m-%d')
            )
            df = df.rename(columns={'date': 'time_counter'})
        df['time_counter'] = pd.to_datetime(df['time_counter'])
        return df

    # Function for merging dataframes from each run
    def merge_dfs(dfs):
        merged = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=['time_counter'],
                how='inner'
            ),
            dfs
        )
        return merged

    # Opening the ANHA4 MLD data
    runs = ['EPM157', 'EPM158', 'EPM151', 'EPM152', 'EPM155', 'EPM156']
    df = []
    for run in runs:
        df_temp = open_processed_data(run)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.set_index('time_counter')
    df = df.loc['2007-12-01':'2017-08-01']
    df = df.loc[(df.index.month > 11) | (df.index.month < 5)]

    # Opening the LAB60 MLD data
    df_ECP017 = open_processed_data(run='LAB60')
    df_ECP017 = df_ECP017.set_index('time_counter')
    df_ECP017 = df_ECP017.loc['2007-12-01':'2017-08-01']
    df_ECP017 = df_ECP017.loc[
        (df_ECP017.index.month > 11) | (df_ECP017.index.month < 5)
    ]

    # Opening the Argo MLD data
    df_argo = open_processed_data(run='Argo')
    df_argo = df_argo.set_index('time_counter')
    df_argo = df_argo.loc['2007-12-01':'2017-08-01']
    df_argo = df_argo.loc[
        (df_argo.index.month > 11) | (df_argo.index.month < 5)
    ]

    # Plotting the MLD bar charts
    means = pd.concat(
        [df.mean(axis=0),
         df_ECP017.mean(axis=0),
         df_argo.mean(axis=0)]
    )
    colours = plt.cm.viridis([0, 0, 0.5, 0.5, 0.8, 0.8, 1, 1])
    hatches = ["", "///", "", "///", "", "///", 'x', 'x']
    means.plot.bar(
        ax=ax1,
        color=colours,
        hatch=hatches,
        width=1,
        edgecolor='w',
        legend=False,
        zorder=100
    )
    ax1.set_ylabel(r'MLD ($m$)', fontdict={'fontsize': 12})
    ax1.set_title(
        r'10-yr winter mean MLD',
        fontdict={'fontsize': 12}
    )
    ax1.set_xticks(
        [0.5, 2.5, 4.5, 6, 7],
        ['Control', 'Tides', 'Tides\nMLEp', '1/60°\nmodel', 'Argo'],
        rotation=0,
        fontsize=12
    )
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.text(
        0.1,
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
    ax1.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=0)

    # Handling the legend
    labels = ['CGRF', 'ERA-I', 'Other']
    hatches = ['', "///", "x"]
    handles = []
    for n, _ in enumerate(labels):
        handles.append(plt.Rectangle((0, 0), 1, 1, fill=0, hatch=hatches[n]))
    ax1.legend(
        handles,
        labels,
        loc='upper right',
        ncol=1,
        fontsize=12,
        shadow=True
    )
    labls = [
        means['EPM157'],
        means['EPM158'],
        means['EPM151'],
        means['EPM152'],
        means['EPM155'],
        means['EPM156'],
        means['LAB60'],
        means['Argo']
    ]
    ax1.set_ylim(bottom=0, top=775)
    kwargs = {'rotation': 0, 'fontsize': 12}
    ax1.bar_label(
        ax1.containers[0],
        labels=[f'{i:.0f}' for i in labls],
        padding=3,
        **kwargs
    )
    for spine in ax1.spines.values():
        spine.set_zorder(120)

    # == Plotting maps == #
    def plt_mini_map(run, title, ax, letter):
        fp = 'MLD_yearly_maps_full_domain_'+run+'.nc'
        ds = xr.open_dataset(fp)
        da = ds['yearly_max'].sel(year=slice(2008, 2017)).mean('year')
        land_50m = feature.NaturalEarthFeature(
            'physical',
            'land',
            '50m',
            edgecolor='black',
            facecolor='gray'
        )
        ax.set_extent(
            [westLon, eastLon, southLat, northLat],
            crs=ccrs.PlateCarree()
        )
        ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
        ax.coastlines(resolution='50m')
        gl = ax.gridlines(
            draw_labels=True,
            dms=False,
            x_inline=False,
            y_inline=False,
            linewidth=0.5
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        gl.rotate_labels = False
        if run == 'EPM157':
            gl.right_labels = True
        if run == 'EPM151':
            gl.right_labels = True
        if run == 'EPM155':
            gl.right_labels = True
        if run == 'EPM158':
            gl.left_labels = True
            gl.bottom_labels = True
        if run == 'EPM152':
            gl.bottom_labels = True
        if run == 'EPM156':
            gl.bottom_labels = True
            gl.right_labels = True
        gl.ylocator = mticker.FixedLocator([50, 55, 60, 65, 70, 75, 80])
        gl.xlocator = mticker.FixedLocator([-45, -55, -65])
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        p = ax.pcolormesh(
            ds.nav_lon_grid_T,
            ds.nav_lat_grid_T,
            da,
            transform=ccrs.PlateCarree(),
            cmap='plasma',
            vmin=0,
            vmax=2500,
            rasterized=True)
        ax.text(
            0.2,
            0.9,
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
        ax.text(
            0.93,
            0.2,
            title,
            transform=ax.transAxes,
            fontsize=12,
            va='top',
            ha='right',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.6
            )
        )
        return p

    plt_mini_map('EPM157', r'CGRF-C', ax2, 'b')
    plt_mini_map('EPM151', r'CGRF-T', ax3, 'c')
    plt_mini_map('EPM155', r'CGRF-TM', ax4, 'd')
    plt_mini_map('EPM158', r'ERAI-C', ax5, 'e')
    plt_mini_map('EPM152', r'ERAI-T', ax6, 'f')
    p = plt_mini_map('EPM156', r'ERAI-TM', ax7, 'g')

    # Adding a colorbar for the maps
    axins = inset_axes(
        ax6,
        width="212%",  # width: 5% of parent_bbox width
        height="10%",  # height: 50%
        loc="upper center",
        bbox_to_anchor=(-0.57, 0.315, 0.95, 1.),
        bbox_transform=ax6.transAxes,
        borderpad=0
    )
    fig.colorbar(p, cax=axins, orientation='horizontal')
    axins.set_xlabel(
        r'10-yr mean of yearly max MLD ($m$)',
        size=12,
        labelpad=8
    )
    axins.xaxis.set_label_position('top')
    axins.yaxis.set_label_coords(-0.08, -1.3)
    axins.tick_params(labelsize=12)

    fig.subplots_adjust(
        bottom=0.06,
        top=0.94,
    )

    name = 'figure_MLDs_body.svg'
    plt.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(name, dpi=600)
    plt.close(fig)

    print("Saved: " + name)


if __name__ == "__main__":
    plot_MLDs_body_figure()
