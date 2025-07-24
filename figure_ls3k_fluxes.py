# For figure of salt and heat flux into the interior Lab Sea across the 3,000 m isobath
# Rowan Brown
# July 2025

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
from ls3k_mask_boundary import ls3k_boundary
import gsw
from cftime import DatetimeNoLeap

def create_temporary_files():
    """The flux datasets are a bit large and slow to open and process every time I want to plot, 
    so this function can be used to create the temporary datasets needed for plotting."""

    # Open the section "mesh" 
    ds_mesh = xr.open_dataset('masks/ls3k_flux_face_hzdims.nc')

    # Identify sections
    irminger_start_id = 75
    irminger_end_id = 120 
    lc_start_id = 180
    lc_end_id = 250

    # For each run save things to plot later
    start_date, end_date = DatetimeNoLeap(2007,12,1), DatetimeNoLeap(2017,11,30)
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
        ds = xr.open_dataset('ls3k_fluxes_'+run+'.nc')
        ds['vol_flux_section_mean'] = ds['vol_flux'].sel(time_counter=slice(start_date, end_date)).mean(['time_counter'])
        ds['heat_flux_section_mean'] = ds['heat_flux'].sel(time_counter=slice(start_date, end_date)).mean(['time_counter'])
        ds['salt_flux_section_mean'] = ds['salt_flux'].sel(time_counter=slice(start_date, end_date)).mean(['time_counter'])
        ds['vol_flux_irminger'] = ds['vol_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])
        ds['heat_flux_irminger'] = ds['heat_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])
        ds['salt_flux_irminger'] = ds['salt_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])
        ds['vol_flux_lc'] = ds['vol_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])
        ds['heat_flux_lc'] = ds['heat_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])
        ds['salt_flux_ls'] = ds['salt_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])
        ds = ds.drop_vars(['vol_flux','heat_flux','salt_flux'])
        ds.to_netcdf('ls3f_fluxes_plotting_'+run+'.nc')
        print("Completed plotting preprocessing for "+run)

def ls3k_plot_fluxes():
    """Creates figure of fluxes for paper."""

    print('Starting figure init')

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1'],
              ['ax2'],
              ['ax3'],
              ['ax4'],
              ['ax5'],
              ['ax6']]
    fig, axd = plt.subplot_mosaic(layout,figsize=(19*cm, 24*cm))
    ax1, ax2, ax3, ax4, ax5, ax6 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4'], axd['ax5'], axd['ax6']

    print('Figure initiated')

    # Opening datasts and taking means
    EPM151 = xr.open_dataset('ls3k_fluxes_plotting_EPM151.nc')
    EPM155 = xr.open_dataset('ls3k_fluxes_plotting_EPM155.nc')
    EPM157 = xr.open_dataset('ls3k_fluxes_plotting_EPM157.nc')

    # Plotting
    ax1.plot.p

    # Saving
    plt.savefig('figure_ls3k_fluxes.pdf',format='pdf',dpi=600)

if __name__=="__main__":
    create_temporary_files()
