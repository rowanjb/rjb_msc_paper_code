# Makes plot of Lab Sea heat and salt content
# Rowan Brown
# August 2025

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from functools import reduce
from datetime import datetime


def contents_figure():
    """Creates plot of heat and salt content in the Lab Sea."""

    print("Beginning: Lab Sea heat and salt content figure")

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
    def open_processed_data(run, fp):
        df = xr.open_dataarray(fp).drop_vars(
            ['time_centered']).to_dataframe(run)
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

    # Opening the salt content data
    df = []
    for run in runs:
        fp = 'ls3k_salt_content_' + run + '.nc'
        df_temp = open_processed_data(run, fp)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.where(df != 0)  # Masking a zero somewher
    df = df/1e18  # Grams to tonnes

    # Plotting the salt content time series
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    df = df.groupby(df.index.shift(
        1, freq='ME').shift(1, freq='d').year).mean()
    df.plot(
        ax=ax1,
        xticks=years,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100
    )
    ax1.set_xlabel(None)
    ax1.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax1.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_tick_params(labelsize=9)
    ax1.set_ylabel('Salt content\n'+r'($\times 10^{12}$ $tonnes$)',
                   fontdict={'fontsize': 12})
    ax1.set_title('')
    ax1.set_ylim(7.422e1, 7.433275e1)
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
    ax1.set_title(
        r'Yearly mean (surface to sea floor)',
        fontdict={'fontsize': 12}
    )

    # Plotting the freshwater content bar chart
    means = df.mean(axis=0)
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
        ['Control', 'Tides', 'Tides\nSMLEp'], rotation=0, fontsize=9)
    ax2.yaxis.set_tick_params(labelsize=9)
    ax2.set_title(
        r'10-yr mean',
        fontdict={'fontsize': 12}
    )
    ax2.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=-10)
    ax2.set_ylim(7.422e1, 7.433275e1)
    labls = [
        means[runs[0]],
        means[runs[1]],
        means[runs[2]],
        means[runs[3]],
        means[runs[4]],
        means[runs[5]]
    ]
    kwargs = {'rotation': 90, 'fontsize': 9,
              'bbox': dict(facecolor='white',
                           edgecolor='none',
                           alpha=0.75,
                           boxstyle='square,pad=0.1')}
    ax2.bar_label(
        ax2.containers[0],
        labels=[str(i)[:6] for i in labls],
        padding=-35,
        zorder=120,
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
            boxstyle='circle,pad=0.1')
    )
    ax2.yaxis.set_ticklabels([])
    for spine in ax2.spines.values():
        spine.set_zorder(110)

    # Opening the heat content data
    df = []
    for run in runs:
        fp = 'ls3k_heat_content_' + run + '.nc'
        df_temp = open_processed_data(run, fp)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.where(df > 1)  # Masking a zero somewher
    df = df/1e21  # J to ZJ

    # Plotting the heat content time series
    df = df.groupby(df.index.shift(
        1, freq='ME').shift(1, freq='d').year).mean()
    df.plot(
        ax=ax3,
        xticks=years,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100
    )
    ax3.set_title('')
    ax3.set_ylabel('Heat content\n'+r'($ZJ$)', fontdict={'fontsize': 12})
    ax3.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.set_xlabel(None)
    ax3.yaxis.set_tick_params(labelsize=9)
    ax3.xaxis.set_tick_params(labelsize=9)
    ax3.set_ylim(47.5, 50.75)
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

    # Plotting the heat content bar chart
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
    ax4.set_title('')
    ax4.set_xticks([0.5, 2.5, 4.5],
                   ['Control', 'Tides', 'Tides\nSMLEp'], rotation=0, fontsize=9)
    ax4.set_ylim(47.5, 50.75)
    ax4.yaxis.set_tick_params(labelsize=9)
    labls = [
        means[runs[0]],
        means[runs[1]],
        means[runs[2]],
        means[runs[3]],
        means[runs[4]],
        means[runs[5]]
    ]
    kwargs = {'rotation': 90, 'fontsize': 9,
              'bbox': dict(facecolor='white',
                           edgecolor='none',
                           alpha=0.75,
                           boxstyle='square,pad=0.1')}
    ax4.bar_label(
        ax4.containers[0],
        labels=[str(i)[:6] for i in labls],
        padding=-35,
        zorder=120,
        **kwargs)
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
    ax4.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=-10)
    ax4.yaxis.set_ticklabels([])
    for spine in ax4.spines.values():
        spine.set_zorder(120)

    # Legends
    handles, labels = ax3.get_legend_handles_labels()
    new_labels = ['GDPS-C', 'ERAI-C', 'GDPS-T', 'ERAI-T', 'GDPS-TS', 'ERAI-TS']
    ax3.legend(
        handles=handles,
        labels=new_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        shadow=True,
        fontsize=9
    )
    labels = ['GDPS', 'ERA-I']
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
        top=0.94,
        left=0.14,
        right=0.96
    )

    name = 'figure_LabSea_HC_and_SC.svg'
    plt.savefig(name, dpi=600)

    print("Saved: "+name)


def contents_figure_400():
    """Creates plot of heat and salt content in the Lab Sea."""

    print("Beginning: Lab Sea heat and salt content figure")

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
    def open_processed_data(run, fp):
        df = xr.open_dataarray(fp).drop_vars(
            ['time_centered']).to_dataframe(run)
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

    # Opening the salt content data
    df = []
    for run in runs:
        fp = 'ls3k_salt_content_' + run + '_400.nc'
        df_temp = open_processed_data(run, fp)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.where(df != 0)  # Masking a zero somewher
    df = df/1e18  # Grams to tonnes

    # Plotting the salt content time series
    years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    df = df.groupby(df.index.shift(
        1, freq='ME').shift(1, freq='d').year).mean()
    df.plot(
        ax=ax1,
        xticks=years,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100
    )
    ax1.set_xlabel(None)
    ax1.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax1.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_tick_params(labelsize=9)
    ax1.set_ylabel('Salt content\n'+r'($\times 10^{12}$ $tonnes$)',
                   fontdict={'fontsize': 12})
    ax1.set_title('')
    ax1.set_ylim(9.16, 9.21)
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
    ax1.set_title(
        r'Yearly mean (surface to 400 m)',
        fontdict={'fontsize': 12}
    )

    # Plotting the freshwater content bar chart
    means = df.mean(axis=0)
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
        ['Control', 'Tides', 'Tides\nSMLEp'], rotation=0, fontsize=9)
    ax2.yaxis.set_tick_params(labelsize=9)
    ax2.set_title(
        r'10-yr mean',
        fontdict={'fontsize': 12}
    )
    ax2.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=-10)
    ax2.set_ylim(9.16, 9.21)
    labls = [
        means[runs[0]],
        means[runs[1]],
        means[runs[2]],
        means[runs[3]],
        means[runs[4]],
        means[runs[5]]
    ]
    kwargs = {'rotation': 90, 'fontsize': 9,
              'bbox': dict(facecolor='white',
                           edgecolor='none',
                           alpha=0.75,
                           boxstyle='square,pad=0.1')}
    ax2.bar_label(
        ax2.containers[0],
        labels=[str(i)[:6] for i in labls],
        padding=-35,
        zorder=120,
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
            boxstyle='circle,pad=0.1')
    )
    ax2.yaxis.set_ticklabels([])
    for spine in ax2.spines.values():
        spine.set_zorder(110)

    # Opening the heat content data
    df = []
    for run in runs:
        fp = 'ls3k_heat_content_' + run + '_400.nc'
        df_temp = open_processed_data(run, fp)
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.where(df > 1)  # Masking a zero somewher
    df = df/1e21  # J to ZJ

    # Plotting the heat content time series
    df = df.groupby(df.index.shift(
        1, freq='ME').shift(1, freq='d').year).mean()
    df.plot(
        ax=ax3,
        xticks=years,
        color=colours,
        style=linestyles,
        legend=False,
        zorder=100
    )
    ax3.set_title('')
    ax3.set_ylabel('Heat content\n'+r'($ZJ$)', fontdict={'fontsize': 12})
    ax3.xaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.yaxis.grid(True, linewidth=1, alpha=0.75)
    ax3.set_xlabel(None)
    ax3.yaxis.set_tick_params(labelsize=9)
    ax3.xaxis.set_tick_params(labelsize=9)
    ax3.set_ylim(6.5, 7.8)
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

    # Plotting the heat content bar chart
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
    ax4.set_title('')
    ax4.set_xticks([0.5, 2.5, 4.5],
                   ['Control', 'Tides', 'Tides\nSMLEp'], rotation=0, fontsize=9)
    ax4.set_ylim(6.5, 7.8)
    ax4.yaxis.set_tick_params(labelsize=9)
    labls = [
        means[runs[0]],
        means[runs[1]],
        means[runs[2]],
        means[runs[3]],
        means[runs[4]],
        means[runs[5]]
    ]
    kwargs = {'rotation': 90, 'fontsize': 9,
              'bbox': dict(facecolor='white',
                           edgecolor='none',
                           alpha=0.75,
                           boxstyle='square,pad=0.1')}
    ax4.bar_label(
        ax4.containers[0],
        labels=[str(i)[:6] for i in labls],
        padding=-35,
        zorder=120,
        **kwargs)
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
    ax4.yaxis.grid(True, linewidth=1, alpha=0.75, zorder=-10)
    ax4.yaxis.set_ticklabels([])
    for spine in ax4.spines.values():
        spine.set_zorder(120)

    # Legends
    handles, labels = ax3.get_legend_handles_labels()
    new_labels = ['GDPS-C', 'ERAI-C', 'GDPS-T', 'ERAI-T', 'GDPS-TS', 'ERAI-TS']
    ax3.legend(
        handles=handles,
        labels=new_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        shadow=True,
        fontsize=9
    )
    labels = ['GDPS', 'ERA-I']
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
        top=0.94,
        left=0.14,
        right=0.96
    )

    name = 'figure_LabSea_HC_and_SC_400.svg'
    plt.savefig(name, dpi=600)

    print("Saved: "+name)


if __name__ == "__main__":
    contents_figure()
    contents_figure_400()
