# Rowan Brown, 13 Aug 2025

import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime


def plot_stratification_figure():
    """Make a plot of convective resistance and volume."""

    print("Beginning: Convective resistance and volume plots")

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

    # Function for opening and processing the data (conv vol and conv R)
    # and storing it in a Pandas dataframe
    def open_processed_data(run, fp):
        ds = xr.open_dataarray(fp)
        try:  # Some of the data has deptht as a coord...
            df = ds.drop_vars(['time_centered', 'deptht']).to_dataframe(run)
        except:  # ...and some doesn't
            df = ds.drop_vars('time_centered').to_dataframe(run)
        df = df.reset_index()
        df['time_counter'] = df['time_counter'].astype(str)
        df['time_counter'] = df['time_counter'].map(
            lambda date_string: datetime.strptime(
                date_string,
                '%Y-%m-%d %H:%M:%S'
            )
        )
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

    # Opening the convective resistance data
    df = []
    for run in runs:
        fp = 'ls3k_convective_resistance_'+run+'.nc'
        df_temp = open_processed_data(run, fp)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.set_index('time_counter')
    # Defining our period of interest
    # Note the years are defined by the winter (including previous December)
    df = df.loc['2007-12-01':'2017-11-30']
    df = df.where(df != 0)  # Masking any spurious zeros
    df = df/1e15  # Handling the units (J -> PJ)

    # Plotting the convective resistance time series
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    years_dt = [datetime(year, 1, 1) for year in years]
    df.plot(
        ax=ax1,
        color=colours,
        xticks=years_dt,
        style=linestyles,
        legend=False,
        zorder=100,
    )
    ax1.set_xticklabels(labels=years, ha='center', rotation=0)
    ax1.set_xlabel(None)
    ax1.xaxis.grid(True, linewidth=1, alpha=0.75)
    means = df.mean(axis=0)
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_tick_params(labelsize=9)
    ax1.set_ylabel(
        'Convective\nresistance ($PJ$)',
        fontdict={'fontsize': 12}
    )
    ax1.set_title(
        r'Mean stratification',
        fontdict={'fontsize': 12}
    )
    ax1.set_ylim(100, 2350)
    ax1.set_xlim(datetime(2007, 7, 2), datetime(2018, 7, 2))
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
    ax1.yaxis.grid(True, linewidth=1, zorder=-10, alpha=0.75)
    for spine in ax1.spines.values():
        spine.set_zorder(120)

    # Plotting the convective resistance bar chart
    means.plot.bar(
        ax=ax2,
        xlabel='',
        color=colours,
        hatch=hatches,
        width=1,
        edgecolor='w',
        label='',
        legend=False,
        ylabel='',
        zorder=100
    )
    ax2.set_xticks(
        [0.5, 2.5, 4.5],
        ['Control', 'Tides', 'Tides\nSMLEs'],
        rotation=0,
        fontsize=9
    )
    ax2.yaxis.set_tick_params(labelsize=9)
    ax2.set_title(
        r'10-yr mean',
        fontdict={'fontsize': 12}
    )
    ax2.set_ylim(100, 2350)
    labls = [
        means['EPM157'],
        means['EPM158'],
        means['EPM151'],
        means['EPM152'],
        means['EPM155'],
        means['EPM156']
    ]
    kwargs = {'rotation': 90, 'fontsize': 9}
    ax2.bar_label(
        ax2.containers[0],
        labels=[str(i)[:6] for i in labls],
        padding=3,
        **kwargs
    )
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
    ax2.yaxis.set_ticklabels([])
    ax2.yaxis.grid(True, linewidth=1, zorder=-10, alpha=0.75)

    # Opening the convective volume data
    df = []
    for run in runs:
        fp = '1kMLD_convective_volume_'+run+'.nc'
        df_temp = open_processed_data(run, fp)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.set_index('time_counter')
    df = df.loc['2007-12-01':'2017-08-01']
    df = df.loc[(df.index.month > 11) | (df.index.month < 5)]
    df = df.replace(0, np.nan)
    df = df/1e12  # /1e9 goes from m3 to km3, /1e12 goes thousand km3

    # Plotting the convective volume time series
    df = df.groupby(df.index.shift(2, freq='ME').year).mean()
    df.plot(
        ax=ax3,
        xticks=years,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100
    )
    ax3.set_ylabel(
        'Winter mean\nconvective\nvolume ($Tm^3$)',
        fontdict={'fontsize': 12}
    )
    ax3.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.set_xlabel(None)
    ax3.yaxis.set_tick_params(labelsize=9)
    ax3.xaxis.set_tick_params(labelsize=9)
    ax3.set_ylim(0, 650)
    ax3.set_xlim(2007.5, 2018.5)
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
    ax3.yaxis.grid(True, linewidth=1, zorder=-10, alpha=0.75)

    # Plotting the convective volume bar chart
    means = pd.concat([df.mean(axis=0)])
    means.plot.bar(
        ax=ax4,
        color=colours,
        hatch=hatches,
        width=1,
        edgecolor='w',
        legend=False,
        ylabel='',
        zorder=100
    )
    ax4.set_xticks(
        [0.5, 2.5, 4.5],
        ['Control', 'Tides', 'Tides\nSMLEs'],
        rotation=0,
        fontsize=9
    )
    ax4.set_ylim(0, 650)
    ax4.yaxis.set_tick_params(labelsize=9)
    labls = [
        means['EPM157'],
        means['EPM158'],
        means['EPM151'],
        means['EPM152'],
        means['EPM155'],
        means['EPM156']
    ]
    kwargs = {'rotation': 90, 'fontsize': 9}  # For annotating mean values
    ax4.bar_label(
        ax4.containers[0],
        labels=[str(i)[:6] for i in labls],
        padding=3,
        **kwargs
    )
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
    ax4.yaxis.set_ticklabels([])
    ax4.yaxis.grid(True, linewidth=1, zorder=-10, alpha=0.75)
    for spine in ax4.spines.values():
        spine.set_zorder(120)

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

    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.05)
    fig.subplots_adjust(
        bottom=0.22,
        top=0.94,
        left=0.14,
        right=0.96
    )

    name = 'figure_ConvR_ConvV.svg'
    plt.savefig(name)
    plt.close(fig)
    print("Saved: "+name)

if __name__ == "__main__":
    plot_stratification_figure()
