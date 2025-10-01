# Looks at oxygen and CO2 contents
# Rowan Brown
# 25 Oct 2023

import pandas as pd
import numpy as np
import xarray as xr
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime


def biogeochem_plot():
    """Makes plots of oxygen and carbon contents."""

    print("Beginning: Biogeochem plot")

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(19*cm, 12*cm),
        gridspec_kw={'width_ratios': [3, 1.05]}
    )

    # For controlling linestyle
    runs = ['EPM157', 'EPM158', 'EPM151', 'EPM152', 'EPM155', 'EPM156']
    colours = plt.cm.viridis([0, 0, 0.5, 0.5, 0.8, 0.8])
    linestyles = ['-', '--', '-', '--', '-', '--']
    hatches = ["", "///", "", "///", "", "///"]

    # Function for opening and processing the data and storing in a pd df
    def open_processed_data(run, fp, var):
        df = xr.open_dataset(fp).drop_vars(
            ['time_centered'])[var].to_dataframe(run)
        df = df.reset_index()
        df['time_counter'] = df['time_counter'].astype(str)
        df['time_counter'] = df['time_counter'].map(
            lambda date_string: datetime.strptime(
                date_string, '%Y-%m-%d %H:%M:%S'))
        df['time_counter'] = pd.to_datetime(df['time_counter'])
        df = df.set_index('time_counter')
        df = df.loc['2007-12-01':'2017-11-30']
        return df

    # Function for merging dataframes from each run
    def merge_dfs(dfs):
        merged = reduce(
            lambda left, right: pd.merge(
                left, right, on=['time_counter'], how='inner'), dfs)
        return merged

    # Carbon
    df_vodic = []
    for run in runs:
        var = 'dic_avg_conc'
        fp = 'ls3k_biogeochem_' + run + '.nc'
        df_tmp = open_processed_data(run, fp, var)
        df_vodic.append(df_tmp)
    df_vodic = merge_dfs(df_vodic)

    # Oxygen
    df_vooxy = []
    for run in runs:
        var = 'ox_avg_conc'
        fp = 'ls3k_biogeochem_' + run + '.nc'
        df_tmp = open_processed_data(run, fp, var)
        df_vooxy.append(df_tmp)
    df_vooxy = merge_dfs(df_vooxy)

    # Carbon time series
    years = np.arange(2007, 2019)
    df_vodic.plot(
        ax=ax1,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100,
    )
    ax1.set_xticklabels(years, rotation=0, ha='center')
    ax1.set_xlabel(None)
    ax1.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax1.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_tick_params(labelsize=9)
    ax1.xaxis.grid(True, linewidth=1, zorder=-10)
    ax1.yaxis.grid(True, linewidth=1, zorder=-10)
    ax1.set_ylim(2.197, 2.22)
    ax1.set_ylabel(
        'DIC\n'+r'($mol$ $m^{-3}$)',
        fontdict={'fontsize': 12}
    )
    ax1.set_title(
        r'Mean concentration',
        fontdict={'fontsize': 12}
    )
    ax1.text(
        0.05,
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

    # Carbon means
    df_vodic_avgs = df_vodic.mean(axis=0, numeric_only=True, skipna=True)
    ax2.set_title('')
    ax2.text(
        0.5,
        1.15,
        '10-yr mean\nconcentration',
        fontsize=12,
        transform=ax2.transAxes,
        ha='center',
        va='center',
    )
    df_vodic_avgs.plot.bar(
        ax=ax2,
        color=colours,
        width=1,
        edgecolor='w',
        hatch=hatches,
        legend=False,
        xlabel='',
        ylabel='',
        zorder=100
    )
    ax2.yaxis.set_tick_params(labelsize=9)
    ax2.xaxis.set_tick_params(labelsize=9)
    ax2.set_xticks(
        [0.5, 2.5, 4.5],
        ['Control', 'Tides', 'Tides\nMLEp'],
        rotation=0
    )
    labls = [
        df_vodic_avgs[runs[0]],
        df_vodic_avgs[runs[1]],
        df_vodic_avgs[runs[2]],
        df_vodic_avgs[runs[3]],
        df_vodic_avgs[runs[4]],
        df_vodic_avgs[runs[5]]
    ]
    kwargs = {'rotation': 90, 'fontsize': 9}
    ax2.bar_label(
        ax2.containers[0],
        labels=[f'{i:.4f}' for i in labls],
        padding=3,
        **kwargs)
    ax2.set_ylim(2.197, 2.22)
    ax2.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=-10)
    ax2.yaxis.set_ticklabels([])
    ax2.text(
        0.15,
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
    for spine in ax2.spines.values():
        spine.set_zorder(110)

    # Oxygen time series
    df_vooxy.plot(
        ax=ax3,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100
    )
    ax3.set_xticklabels(years, rotation=0, ha='center')
    ax3.set_xlabel(None)
    ax3.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.xaxis.set_tick_params(labelsize=9)
    ax3.yaxis.set_tick_params(labelsize=9)
    ax3.xaxis.grid(True, linewidth=1, zorder=-10)
    ax3.yaxis.grid(True, linewidth=1, zorder=-10)
    ax3.set_ylim(0.2725, 0.29)
    ax3.set_ylabel(
        'Oxygen\n'+r'($mol$ $m^{-3}$)',
        fontdict={'fontsize': 12}
    )
    ax3.set_title('')
    ax3.text(
        0.05,
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

    # Oxygen means
    df_vooxy_avgs = df_vooxy.mean(axis=0, numeric_only=True, skipna=True)
    ax4.set_title('')
    df_vooxy_avgs.plot.bar(
        ax=ax4,
        color=colours,
        width=1,
        edgecolor='w',
        hatch=hatches,
        legend=False,
        xlabel='',
        ylabel='',
        zorder=100
    )
    ax4.yaxis.set_tick_params(labelsize=9)
    ax4.xaxis.set_tick_params(labelsize=9)
    ax4.set_xticks(
        [0.5, 2.5, 4.5],
        ['Control', 'Tides', 'Tides\nMLEp'],
        rotation=0
    )
    labls = [
        df_vooxy_avgs[runs[0]],
        df_vooxy_avgs[runs[1]],
        df_vooxy_avgs[runs[2]],
        df_vooxy_avgs[runs[3]],
        df_vooxy_avgs[runs[4]],
        df_vooxy_avgs[runs[5]]
    ]
    kwargs = {'rotation': 90, 'fontsize': 9}
    ax4.bar_label(
        ax4.containers[0],
        labels=[f'{i:.4f}' for i in labls],
        padding=3,
        **kwargs)
    ax4.set_ylim(0.2725, 0.29)
    ax4.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=-10)
    ax4.yaxis.set_ticklabels([])
    ax4.text(
        0.15,
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
        spine.set_zorder(110)

    # Legends
    handles, labels = ax3.get_legend_handles_labels()
    new_labels = ['CGRF-C', 'ERAI-C', 'CGRF-T', 'ERAI-T', 'CGRF-TS', 'ERAI-TS']
    ax3.legend(
        handles=handles,
        labels=new_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        shadow=True,
        fontsize=9
    )
    labels = ['CGRF', 'ERA-I']
    hatches = ['', "///"]
    handles = []
    for n, _ in enumerate(labels):
        handles.append(plt.Rectangle((0, 0), 1, 1, fill=0, hatch=hatches[n]))
    ax4.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        shadow=True,
        fontsize=9
    )

    plt.subplots_adjust(
        hspace=0.3,
        wspace=0.05,
        bottom=0.22,
        top=0.90,
        left=0.14,
        right=0.96
    )

    name = 'figure_biogeochem.svg'
    plt.savefig(name, dpi=600)

    print("Saved: " + name)


if __name__ == '__main__':
    biogeochem_plot()
