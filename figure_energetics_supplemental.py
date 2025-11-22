# Python script for creating plots relating to energetics in the Lab Sea

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import colors
from matplotlib.collections import PolyCollection
from datetime import datetime
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as feature
from cftime import DatetimeNoLeap
import matplotlib as mpl
import matplotlib.ticker as mticker
import pyproj


def energetics_plot():
    """Creates violin plot of energetics..."""

    print("Beginning: energetics supplemental figure")

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1'],
              ['ax2'],
              ['ax3'],
              ['ax4'],
              ['ax5']]
    fig, axd = plt.subplot_mosaic(layout, figsize=(15*cm, 19*cm))
    ax1, ax2, ax3, ax4 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4']
    ax5 = axd['ax5']

    # Function for merging dataframes
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

    # Plotting in a loop!
    axes = [ax1, ax2, ax3, ax4, ax5]
    runs = ['EPM157', 'EPM158', 'EPM151', 'EPM152', 'EPM155', 'EPM156']
    names = [' ', 'GDPS-C', 'ERAI-C', 'GDPS-T', 'ERAI-T', 'GDPS-TM', 'ERAI-TM']
    dtype = ['EKE', 'T1', 'T2', 'T3', 'T4']
    # ls = ['-', '--', '-', '--', '-', '--']
    dtype_long = [
        'EKE\n'+r'($TJ$)',
        'MAPE\n'+r'$\downarrow$'+'\nMKE\n'+r'($TW$)',
        'MAPE\n'+r'$\downarrow$'+'\nEAPE\n'+r'($GW$)',  # BC
        'EAPE\n'+r'$\downarrow$'+'\nEKE\n'+r'($GW$)',
        'MKE\n'+r'$\downarrow$'+'\nEKE\n'+r'($MW$)',  # BT
    ]
    letters = ['a', 'b', 'c', 'd', 'e']
    unit_factor = [1e12, 1e12, 1e9, 1e9, 1e6]
    colours = ['r', 'b']
    ylims = [(-0.1e3, 1.2e3),
             (-3.9e1, 3.2e1),
             (-1.1e3, 1.1e3),
             (-0.2e0, 3e0),
             (-2e2, 2e2)]
    for n, ax in enumerate(axes):
        for nd, d in enumerate(['400', '2000']):
            df_list = []
            for nrun, run in enumerate(runs):
                fp = (dtype[n]+'_time_series_ls3k_'+run+
                      '_depth'+d+'m_window21.nc')
                da = xr.open_dataarray(fp)
                da = da.sel(time_counter=slice(
                    DatetimeNoLeap(2008, 1, 1),
                    DatetimeNoLeap(2018, 1, 1)
                    )
                )
                da = da.where(da != 0, drop=True)
                try:
                    da = da.drop_vars('time_centered')
                except ValueError:
                    da = da
                df = da.to_dataframe(run)
                df = df.reset_index()
                df['time_counter'] = df['time_counter'].astype(str)
                df['time_counter'] = df['time_counter'].map(
                    lambda date_string: datetime.strptime(
                        date_string,
                        '%Y-%m-%d %H:%M:%S'
                    )
                )
                df['time_counter'] = pd.to_datetime(df['time_counter'])
                df = df.set_index('time_counter')
                df_list.append(da.to_dataframe(run))
            df = merge_dfs(df_list)
            df = df/unit_factor[n]
            p = ax.violinplot(
                df,
                showextrema=False,
                showmeans=True,
                widths=0.9
            )
            for npc, pc in enumerate(p['bodies']):
                pc.set_facecolor('none')
                pc.set_edgecolor(colours[nd])
                pc.set_linewidth(1.5)
                pc.set_alpha(0.7)
                # pc.set_linestyle(ls[npc])
            p['cmeans'].set_colors(colours[nd])
            ax.set_ylim(ylims[n])
            ax.text(
                0.05,
                0.95,
                letters[n],
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
            ax.tick_params(axis='both', labelsize=9)
            ax.xaxis.grid(True, linewidth=1, alpha=0.75)
            ax.yaxis.grid(True, linewidth=1, alpha=0.75)
            ax.set_xticklabels(names)
            ax.text(
                -0.16,
                0.5,
                dtype_long[n],
                fontsize=10,
                transform=ax.transAxes,
                va='center',
                ha='center'
            )

    # X labels and title
    ax1.tick_params(top=True, labeltop=True)
    ax1.text(
        0.5,
        1.4,
        'Energetics and conversion rates',
        transform=ax1.transAxes,
        fontsize=12,
        ha='center',
        va='bottom'
    )

    # Legend
    labels = [' 0-400 m  ', ' 400 m-2,000 m']
    lines = [
        plt.Rectangle((0, 0), 1, 1, fill=True, edgecolor=colours[0],
                      facecolor='none', alpha=0.7, lw=1.5),
        plt.Rectangle((0, 0), 1, 1, fill=True, edgecolor=colours[1],
                      facecolor='none', alpha=0.7, lw=1.5),
    ]
    legendLC = plt.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, -0.61),
        title='Depth range',
        ncol=2,
        loc="center",
        fontsize=9,
        shadow=True,
        handleheight=1.625,
        handlelength=3,
    )

    plt.subplots_adjust(
        hspace=0.1,
        wspace=0.2,
        right=0.95,
        left=0.2,
        top=0.9,
        bottom=0.15
    )

    name = 'figure_energetics_supplemental.svg'
    plt.savefig(name, dpi=600)
    print("Saved: "+name)


if __name__ == "__main__":
    energetics_plot()
