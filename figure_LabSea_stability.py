

import matplotlib.pyplot as plt
import xarray as xr


def stability_figure():

    print("Beginning: ")

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(
        nrows=1, ncols=4, figsize=(19*cm, 12*cm))

    # Colours
    c1 = (153/256, 0/256, 2/256)
    c2 = (196/256, 121/256, 0/256)
    c3 = (112/256, 160/256, 205/256)

    # Function for doing the plotting
    def plot_profiles(ds1, ds2, ax, letter, title):
        ds_diff = ds1-ds2
        ds_diff = ds_diff*1E6  # Handling formatting
        ds_diff['N2'].plot(y='deptht', ax=ax, c=c3, label='$N2$')
        ds_diff['N2_temp'].plot(y='deptht', ax=ax, c=c1, label='$N^2_{heat}$')
        ds_diff['N2_salt'].plot(y='deptht', ax=ax, c=c2, label='$N^2_{salt}$')
        ax.invert_yaxis()
        ax.set_ylim(1000, 0)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.xaxis.grid(True, linewidth=1, alpha=0.75)
        ax.yaxis.grid(True, linewidth=1, alpha=0.75)
        ax.xaxis.set_tick_params(labelsize=9)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-1.4, 1.4)
        ax.text(
            0.08,
            0.97,
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
        return

    # Opening the files
    def op_stab(run):
        ds = xr.open_dataset('stability_LS3000_'+run+'.nc')
        ds = ds.where(ds['deptht'] < 1000, drop=True)
        ds_mean = ds.where(ds['time_counter'].dt.month.isin([12, 1, 2, 3, 4]), drop=True).mean("time_counter")
        return ds_mean

    # Plotting
    plot_profiles(op_stab("EPM151"), op_stab("EPM157"), ax1, 'a', 'GDPS: Anomaly\ndue to tides')
    plot_profiles(op_stab("EPM155"), op_stab("EPM151"), ax2, 'b', 'GDPS: Anomaly\ndue to SMLEp')
    plot_profiles(op_stab("EPM152"), op_stab("EPM158"), ax3, 'c', 'ERA-I: Anomaly\ndue to Tides')
    plot_profiles(op_stab("EPM156"), op_stab("EPM152"), ax4, 'd', 'ERA-I: Anomaly\ndue to SMLEp')

    # Misc. formatting
    plt.text(
        0.5, 0.02,
        r"Brunt-Väisälä frequency, $N^2$ ($\times 10^{-6}$ $s^{-1}$)",
        ha='center', va='bottom', transform=fig.transFigure, fontsize=12)
    for ax in [ax2, ax3, ax4]:
        ax.set_yticklabels([])
    ax1.set_ylabel('Depth ($m$)', fontdict={'fontsize': 12})
    ax4.legend(
        loc='lower left',
        bbox_to_anchor=(-2.14, 0.021),
        ncol=3,
        shadow=True,
        fontsize=9
    )
    plt.suptitle("Stability anomalies", fontsize=12)

    plt.subplots_adjust(
        wspace=0.15, bottom=0.15, top=0.825, left=0.12, right=0.96)

    name = 'figure_LabSea_stability.svg'
    plt.savefig(name, dpi=600)


if __name__ == "__main__":
    stability_figure()
