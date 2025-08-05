# For figure of salt and heat flux into the interior Lab Sea across the 3,000 m isobath
# Rowan Brown
# July 2025

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import cartopy.crs as ccrs
import cartopy.feature as feature
from ls3k_mask_boundary import ls3k_boundary
import gsw
from cftime import DatetimeNoLeap
from datetime import datetime,timedelta
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

def create_temporary_files():
    """The flux datasets are a bit large and slow to open and process every time I want to plot, 
    so this function can be used to create the temporary datasets needed for plotting."""

    # Open the section "mesh" 
    ds_mesh = xr.open_dataset('masks/ls3k_flux_face_hzdims.nc')

    # Identify sections
    irminger_start_id = 65
    irminger_end_id = 120 
    lc_start_id = 180
    lc_end_id = 230

    # Set the definition of surface and depth
    d = 300

    # For each run save things to plot later
    start_date, end_date = DatetimeNoLeap(2007,12,1), DatetimeNoLeap(2017,11,30)
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
        
        # Open
        ds = xr.open_dataset('ls3k_fluxes_'+run+'.nc')
       
        #== For making sections --- these aren't that instructive because they're too busy ==#
        ds['vol_flux_section_mean'] = ds['vol_flux'].sel(time_counter=slice(start_date, end_date)).mean(['time_counter'])
        ds['heat_flux_section_mean'] = ds['heat_flux'].sel(time_counter=slice(start_date, end_date)).mean(['time_counter'])
        ds['salt_flux_section_mean'] = ds['salt_flux'].sel(time_counter=slice(start_date, end_date)).mean(['time_counter'])
       
        #== Look at Iringer fluxes --- total, surface, and depth ==#

        # Irminger total
        ds['vol_flux_irminger'] = ds['vol_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])
        ds['heat_flux_irminger'] = ds['heat_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])
        ds['salt_flux_irminger'] = ds['salt_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])
        ds['fw_flux_irminger'] = ds['fw_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).sum(['ids','deptht'])

        # Irminger surface
        ds['vol_flux_irminger_srfc'] = ds['vol_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])
        ds['heat_flux_irminger_srfc'] = ds['heat_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])
        ds['salt_flux_irminger_srfc'] = ds['salt_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])
        ds['fw_flux_irminger_srfc'] = ds['fw_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])

        # Irminger depth
        ds['vol_flux_irminger_depth'] = ds['vol_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])
        ds['heat_flux_irminger_depth'] = ds['heat_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])
        ds['salt_flux_irminger_depth'] = ds['salt_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])
        ds['fw_flux_irminger_depth'] = ds['fw_flux'].sel(ids=slice(irminger_start_id, irminger_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])

        #== Now look at Labrador Current fluxes --- total, surface, and depth ==#

        # LC total
        ds['vol_flux_lc'] = ds['vol_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])
        ds['heat_flux_lc'] = ds['heat_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])
        ds['salt_flux_lc'] = ds['salt_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])
        ds['fw_flux_lc'] = ds['fw_flux'].sel(ids=slice(lc_start_id, lc_end_id)).sum(['ids','deptht'])

        # LC surface
        ds['vol_flux_lc_srfc'] = ds['vol_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])
        ds['heat_flux_lc_srfc'] = ds['heat_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])
        ds['salt_flux_lc_srfc'] = ds['salt_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])
        ds['fw_flux_lc_srfc'] = ds['fw_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']<d).sum(['ids','deptht'])

        # LC depth
        ds['vol_flux_lc_depth'] = ds['vol_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])
        ds['heat_flux_lc_depth'] = ds['heat_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])
        ds['salt_flux_lc_depth'] = ds['salt_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])
        ds['fw_flux_lc_depth'] = ds['fw_flux'].sel(ids=slice(lc_start_id, lc_end_id)).where(ds['deptht']>d).sum(['ids','deptht'])

        #== Finally consider full boundary current fluxes --- total, surface, and depth ==#
        
        # LC total
        ds['vol_flux_full'] = ds['vol_flux'].sum(['ids','deptht'])
        ds['heat_flux_full'] = ds['heat_flux'].sum(['ids','deptht'])
        ds['salt_flux_full'] = ds['salt_flux'].sum(['ids','deptht'])
        ds['fw_flux_full'] = ds['fw_flux'].sum(['ids','deptht'])

        # LC surface
        ds['vol_flux_full_srfc'] = ds['vol_flux'].where(ds['deptht']<d).sum(['ids','deptht'])
        ds['heat_flux_full_srfc'] = ds['heat_flux'].where(ds['deptht']<d).sum(['ids','deptht'])
        ds['salt_flux_full_srfc'] = ds['salt_flux'].where(ds['deptht']<d).sum(['ids','deptht'])
        ds['fw_flux_full_srfc'] = ds['fw_flux'].where(ds['deptht']<d).sum(['ids','deptht'])

        # LC depth
        ds['vol_flux_full_depth'] = ds['vol_flux'].where(ds['deptht']>d).sum(['ids','deptht'])
        ds['heat_flux_full_depth'] = ds['heat_flux'].where(ds['deptht']>d).sum(['ids','deptht'])
        ds['salt_flux_full_depth'] = ds['salt_flux'].where(ds['deptht']>d).sum(['ids','deptht'])
        ds['fw_flux_full_depth'] = ds['fw_flux'].where(ds['deptht']>d).sum(['ids','deptht'])

        #== (Also consider full boundary current fluxes MINUS Irminger and Lab Current sections) ==#

        # LC total
        ds['vol_flux_other'] = ds['vol_flux_full'] - ds['vol_flux_irminger'] - ds['vol_flux_lc']
        ds['heat_flux_other'] = ds['heat_flux_full'] - ds['heat_flux_irminger'] - ds['heat_flux_lc']
        ds['salt_flux_other'] = ds['salt_flux_full'] - ds['salt_flux_irminger'] - ds['salt_flux_lc']
        ds['fw_flux_other'] = ds['fw_flux_full'] - ds['fw_flux_irminger'] - ds['fw_flux_lc']

        # LC surface
        ds['vol_flux_other_srfc'] = ds['vol_flux_full_srfc'] - ds['vol_flux_irminger_srfc'] - ds['vol_flux_lc_srfc'] 
        ds['heat_flux_other_srfc'] = ds['heat_flux_full_srfc'] - ds['heat_flux_irminger_srfc'] - ds['heat_flux_lc_srfc']
        ds['salt_flux_other_srfc'] = ds['salt_flux_full_srfc'] - ds['salt_flux_irminger_srfc'] - ds['salt_flux_lc_srfc']
        ds['fw_flux_other_srfc'] = ds['fw_flux_full_srfc'] - ds['fw_flux_irminger_srfc'] - ds['fw_flux_lc_srfc']

        # LC depth
        ds['vol_flux_other_depth'] = ds['vol_flux_full_depth'] - ds['vol_flux_irminger_depth'] - ds['vol_flux_lc_depth']
        ds['heat_flux_other_depth'] = ds['heat_flux_full_depth'] - ds['heat_flux_irminger_depth'] - ds['heat_flux_lc_depth']
        ds['salt_flux_other_depth'] = ds['salt_flux_full_depth'] - ds['salt_flux_irminger_depth'] - ds['salt_flux_lc_depth']
        ds['fw_flux_other_depth'] = ds['fw_flux_full_depth'] - ds['fw_flux_irminger_depth'] - ds['fw_flux_lc_depth']

        ds = ds.drop_vars(['vol_flux','heat_flux','salt_flux','fw_flux'])
        ds.to_netcdf('ls3k_fluxes_plotting_'+run+'.nc')
        print("Completed plotting preprocessing for "+run)

def test_plot_ls3k_region_defs():
    """Throw-away script showing the region definitions that I'm using."""

    ds_flux = xr.open_dataset('ls3k_fluxes_plotting_EPM152.nc')
    ds = xr.open_dataset('masks/ls3k_flux_mask.nc')
    ds_mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc')
    westLon, eastLon, northLat, southLat = -65, -40, 67, 51
    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='gray')
    projection = ccrs.AlbersEqualArea(central_longitude=-55, central_latitude=50,standard_parallels=(southLat,northLat))
    ax = plt.subplot(1, 1, 1, projection=projection)
    ax.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.coastlines(resolution='50m')
    ax.pcolormesh(ds_mesh.nav_lon,ds_mesh.nav_lat,ds['westward_cells']+ds['eastward_cells']+ds['northward_cells']+ds['southward_cells'],transform=ccrs.PlateCarree())
    ax.scatter(ds_flux['nav_lon_grid_T'].isel(ids=65),ds_flux['nav_lat_grid_T'].isel(ids=65),transform=ccrs.PlateCarree())
    ax.scatter(ds_flux['nav_lon_grid_T'].isel(ids=120),ds_flux['nav_lat_grid_T'].isel(ids=120),transform=ccrs.PlateCarree())
    ax.scatter(ds_flux['nav_lon_grid_T'].isel(ids=180),ds_flux['nav_lat_grid_T'].isel(ids=180),transform=ccrs.PlateCarree())
    ax.scatter(ds_flux['nav_lon_grid_T'].isel(ids=230),ds_flux['nav_lat_grid_T'].isel(ids=230),transform=ccrs.PlateCarree())
    plt.savefig('test.png',dpi=600)

def ls3k_plot_barh_diffs():
    """Creates figure of fluxes for paper."""

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1','ax1','ax1','ax1'],
              ['ax1','ax1','ax1','ax1'],
              ['.'  ,'.'  ,'.'  ,'.'  ],
              ['ax2','ax2','ax2','ax2'],
              ['ax2','ax2','ax2','ax2'],
              ['.'  ,'.'  ,'.'  ,'.'  ],
              ['ax3','ax3','ax3','ax3'],
              ['ax3','ax3','ax3','ax3'],
              ['.'  ,'.'  ,'.'  ,'.'  ],
              ['.'  ,'.'  ,'ax4','ax4'],
              ['.'  ,'.'  ,'ax4','ax4'],
              ['.'  ,'.'  ,'ax4','ax4']]
    westLon, eastLon, northLat, southLat = -65, -40, 67, 51
    projection = ccrs.AlbersEqualArea(central_longitude=-55, central_latitude=50,standard_parallels=(southLat,northLat))
    fig, axd = plt.subplot_mosaic(layout,figsize=(19*cm, 17*cm), per_subplot_kw={("ax4"): {"projection": projection}})
    ax1, ax2, ax3, ax4 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4']

    # Opening datasts and taking means
    EPM151 = xr.open_dataset('ls3k_fluxes_plotting_EPM151.nc')
    EPM152 = xr.open_dataset('ls3k_fluxes_plotting_EPM152.nc')
    EPM155 = xr.open_dataset('ls3k_fluxes_plotting_EPM155.nc')
    EPM156 = xr.open_dataset('ls3k_fluxes_plotting_EPM156.nc')
    EPM157 = xr.open_dataset('ls3k_fluxes_plotting_EPM157.nc')
    EPM158 = xr.open_dataset('ls3k_fluxes_plotting_EPM158.nc')

    #== Bar charts ==#

    # Objects to loop over
    bar_labels = ("Tides (CGRF)" , "Tides (ERA-I)", "MLEp (CGRF)"  , "MLEp (ERA-I)")
    runs       = [(EPM151,EPM157), (EPM152,EPM158), (EPM155,EPM151), (EPM156,EPM152)]

    # Function to create an arracy for the stack
    def stack_value_constructor(stack_key, var, runs):
        ds_var_dict = {
                    "Irminger surface":"_flux_irminger_srfc",
                    "Irminger depth": "_flux_irminger_depth",
                    "Labrador surface": "_flux_lc_srfc",
                    "Labrador depth": "_flux_lc_depth",
                    "Other surface": "_flux_other_srfc",
                    "Other depth": "_flux_other_depth",
                }
        stack_value = np.zeros(len(runs))
        for n,ds in enumerate(runs):
            stack_value[n] = ds[0][ var + ds_var_dict[stack_key]].mean().values - ds[1][ var + ds_var_dict[stack_key]].mean().values
        return stack_value

    # Create the datasets (dicts) containing the individual stacks within each bar
    def populate_stacks(var, runs):
        stack = {
                    "Irminger surface": stack_value_constructor("Irminger surface", var, runs),
                    "Irminger depth":   stack_value_constructor("Irminger depth", var, runs),
                    "Labrador surface": stack_value_constructor("Labrador surface", var, runs),
                    "Labrador depth":   stack_value_constructor("Labrador depth", var, runs),
                    "Other surface":    stack_value_constructor("Other surface", var, runs),
                    "Other depth":      stack_value_constructor("Other depth", var, runs),
                }
        return stack

    # ax1 : Volume flux
    vol_stacks = populate_stacks('vol', runs)
    bottom_pos_neg = np.zeros((2,4)) # Since some fluxes are positive and some negative, we need two tallies
    bottom = np.zeros(4) # This is what we'll pass to the plotting function for every region/key
    cmap = mpl.colormaps['plasma']
    c = cmap(np.linspace(0, 1, 2*len(vol_stacks)))
    colours = {"Irminger surface": c[0], "Irminger depth": c[2], "Labrador surface": c[5],
               "Labrador depth": c[6], "Other surface": c[10], "Other depth": c[11]}
    for region, fluxes in vol_stacks.items(): # For each region...
        fluxes = fluxes/(1000000) # m**3/s -> sv/s
        for n,flux in enumerate(fluxes): # Loop through each "flux", which is really each individual run
            if flux > 0: # If the flux is positive, then...
                bottom[n] = bottom_pos_neg[0][n] # The bottom for this iteration is the previous positive tally's result
                bottom_pos_neg[0][n] = bottom_pos_neg[0][n] + flux # And then we add to the positive top row
            else: # If the flux is negative...
                bottom[n] = bottom_pos_neg[1][n] # And then we can provide this iteration's bottom coordinate
                bottom_pos_neg[1][n] = bottom_pos_neg[1][n] + flux # Then we first need to subtract the current iteration's flux 
                #bottom[n] = bottom_pos_neg[1][n] # And then we can provide this iteration's bottom coordinate
        p = ax1.barh(bar_labels, fluxes, 1, label=region, left=bottom, color=colours[region])#    da1.plot.pcolormesh(x='ids',y='deptht',ax=ax2,rasterized=True)
    ax1.set_title("Differences in 10-yr mean volume fluxes ($Sv$)", pad=-10, fontdict={'fontsize':12})
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_tick_params(labelsize=9)
    ax1.text(0.05, 0.9, 'a', transform=ax1.transAxes,fontsize=14, fontweight='bold', va='top', ha='right',bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.1'))
    ax1.set_xlim(-5.2,5.2)
    
    # ax2 : FW or salt flux
    salt_stacks = populate_stacks('salt', runs)
    bottom_pos_neg = np.zeros((2,4)) # Since some fluxes are positive and some negative, we need two tallies
    bottom = np.zeros(4) # This is what we'll pass to the plotting function for every region/key
    colours = {"Irminger surface": c[0], "Irminger depth": c[2], "Labrador surface": c[5],
               "Labrador depth": c[6], "Other surface": c[10], "Other depth": c[11]}
    for region, fluxes in salt_stacks.items(): # For each region...
        fluxes = fluxes/(1000*1000*1000) # g/s -> k Tonnes/s
        for n,flux in enumerate(fluxes): # Loop through each "flux", which is really each individual run
            if flux > 0: # If the flux is positive, then...
                bottom[n] = bottom_pos_neg[0][n] # The bottom for this iteration is the previous positive tally's result
                bottom_pos_neg[0][n] = bottom_pos_neg[0][n] + flux # And then we add to the positive top row
            else: # If the flux is negative...
                bottom[n] = bottom_pos_neg[1][n] # And then we can provide this iteration's bottom coordinate
                bottom_pos_neg[1][n] = bottom_pos_neg[1][n] + flux # Then we first need to subtract the current iteration's flux 
                #bottom[n] = bottom_pos_neg[1][n] # And then we can provide this iteration's bottom coordinate
        p = ax2.barh(bar_labels, fluxes, 0.9, label=region, left=bottom, color=colours[region])#    da1.plot.pcolormesh(x='ids',y='deptht',ax=ax2,rasterized=True)
    ax2.set_title("Differences in 10-yr mean salt flux ($kt$ $s^{-1}$)", pad=-10,fontdict={'fontsize':12})
    ax2.xaxis.set_tick_params(labelsize=9)
    ax2.yaxis.set_tick_params(labelsize=9)
    ax2.text(0.05, 0.9, 'b', transform=ax2.transAxes,fontsize=14, fontweight='bold', va='top', ha='right',bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.1'))
    ax2.set_xlim(-190,190)

    # ax3 : Heat flux
    heat_stacks = populate_stacks('heat', runs)
    bottom_pos_neg = np.zeros((2,4)) # Since some fluxes are positive and some negative, we need two tallies
    bottom = np.zeros(4) # This is what we'll pass to the plotting function for every region/key
    colours = {"Irminger surface": c[0], "Irminger depth": c[2], "Labrador surface": c[5],
               "Labrador depth": c[6], "Other surface": c[10], "Other depth": c[11]}
    for region, fluxes in heat_stacks.items(): # For each region...
        fluxes = fluxes/(1000000000000) # W -> TW
        for n,flux in enumerate(fluxes): # Loop through each "flux", which is really each individual run
            if flux > 0: # If the flux is positive, then...
                bottom[n] = bottom_pos_neg[0][n] # The bottom for this iteration is the previous positive tally's result
                bottom_pos_neg[0][n] = bottom_pos_neg[0][n] + flux # And then we add to the positive top row
            else: # If the flux is negative...
                bottom[n] = bottom_pos_neg[1][n] # And then we can provide this iteration's bottom coordinate
                bottom_pos_neg[1][n] = bottom_pos_neg[1][n] + flux # Then we first need to subtract the current iteration's flux 
                #bottom[n] = bottom_pos_neg[1][n] # And then we can provide this iteration's bottom coordinate
        p = ax3.barh(bar_labels, fluxes, 0.9, label=region, left=bottom, color=colours[region])#    da1.plot.pcolormesh(x='ids',y='deptht',ax=ax2,rasterized=True)
    ax3.set_title("Differences in 10-yr mean heat flux ($TW$)", pad=-10, fontdict={'fontsize':12})
    ax3.xaxis.set_tick_params(labelsize=9)
    ax3.yaxis.set_tick_params(labelsize=9)
    ax3.text(0.05, 0.9, 'c', transform=ax3.transAxes,fontsize=14, fontweight='bold', va='top', ha='right',bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.1'))
    ax3.set_xlim(-150,150)

    # ax4 : Add map
    lons = np.zeros(len(EPM151['ids']))
    lats = np.zeros(len(EPM151['ids']))
    colours = [None] * len(EPM151['ids'])
    for n,i in enumerate(EPM151['ids'].values):
        lons[n] = EPM151['nav_lon_grid_T'].isel(ids=n).values
        lats[n] = EPM151['nav_lat_grid_T'].isel(ids=n).values
        if i < 66:
            colours[n] = c[10]
        elif i < 121:
            colours[n] = c[0]
        elif i < 181: 
            colours[n] = c[10]
        elif i < 231:
            colours[n] = c[5] 
        else:
            colours[n] = c[10]
    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='gray')
    ax4.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax4.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax4.coastlines(resolution='50m')
    ax4.scatter(lons,lats,s=0.5,c=colours,transform=ccrs.PlateCarree())
    gl = ax4.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.5)
    gl.top_labels=False
    gl.left_labels=False 
    gl.rotate_labels=False
    gl.ylocator = mticker.FixedLocator([50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-45, -55, -65]) 
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax4.text(0.2, 0.9, 'd', transform=ax4.transAxes,fontsize=14, fontweight='bold', va='top', ha='right',bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.1'))

    # Generating some final text to manipulate in inkscape
    ax4.text(0, 0, 'Labrador Current\nsection', transform=ax4.transAxes,fontsize=9)
    ax4.text(0, 0.5, 'West Greenland\nCurrent section', transform=ax4.transAxes,fontsize=9)

    # Legend
    linesLC = [ 
                Line2D([0], [0], color=c[5], lw=14), 
                Line2D([0], [0], color=c[6], lw=14),
              ]
    linesI = [
                Line2D([0], [0], color=c[0], lw=14), 
                Line2D([0], [0], color=c[2], lw=14),
             ]
    linesO = [
                Line2D([0], [0], color=c[10], lw=14), 
                Line2D([0], [0], color=c[11], lw=14)
             ]   
    labels = [' 0-300 m  ',' 300 m-bottom']

    legendLC = plt.legend(linesLC, labels, bbox_to_anchor=(-1.75,0.95), title='Labrador Current fluxes', ncol=2, loc="center", fontsize=9, framealpha=0)
    legendI  = plt.legend(linesI, labels, bbox_to_anchor=(-1.75,0.475), title='West Greenland Current fluxes', ncol=2, loc="center", fontsize=9, framealpha=0)
    legendO  = plt.legend(linesO, labels, bbox_to_anchor=(-1.75,0), title='Other fluxes', ncol=2, loc="center", fontsize=9, framealpha=0)

    ax4.add_artist(legendLC)
    ax4.add_artist(legendI)
    ax4.add_artist(legendO)

    # Saving
    plt.savefig('figure_ls3k_fluxes.pdf',format='pdf',dpi=600)

def ls3k_plot_volume_fluxes():
    """Creates figure of fluxes for paper."""

    areas = xr.open_dataset('masks/ls3k_flux_face_hzdims.nc')['projected_area']
    EPM151 = xr.open_dataset('ls3k_fluxes_plotting_EPM151.nc')
    EPM152 = xr.open_dataset('ls3k_fluxes_plotting_EPM152.nc')
    EPM155 = xr.open_dataset('ls3k_fluxes_plotting_EPM155.nc')
    EPM156 = xr.open_dataset('ls3k_fluxes_plotting_EPM156.nc')
    EPM157 = xr.open_dataset('ls3k_fluxes_plotting_EPM157.nc')
    EPM158 = xr.open_dataset('ls3k_fluxes_plotting_EPM158.nc')

    # Init the figure
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1','ax1','ax1','.'],
              ['ax1','ax1','ax1','.'],
              ['.'  ,'.'  ,'.'  ,'.'],
              ['ax2','ax2','ax2','.'],
              ['ax2','ax2','ax2','.'],
              ['.'  ,'.'  ,'.'  ,'.'],
              ['ax3','ax3','ax3','.'],
              ['ax3','ax3','ax3','.']]
    westLon, eastLon, northLat, southLat = -65, -40, 67, 51
    projection = ccrs.AlbersEqualArea(central_longitude=-55, central_latitude=50,standard_parallels=(southLat,northLat))
    fig, axd = plt.subplot_mosaic(layout,figsize=(19*cm, 19*cm))
    ax1, ax2, ax3 = axd['ax1'], axd['ax2'], axd['ax3']

    # Ax 1: Time series

    # Dictionary of legend entries
    legend_dict = {
        'EPM151': 'C tides',
        'EPM152': 'E tides',
        'EPM155': 'C tides + SMLEs',
        'EPM156': 'E tides + SMLEs',
        'EPM157': 'C control',
        'EPM158': 'E control'
    }

    # For controlling linestyle 
    c1, c2, c3, c4, c5, c6 = plt.cm.viridis([0, 0.5, 0.8, 0, 0.5, 0.8])
    runs = [EPM157,EPM158,EPM151,EPM152,EPM155,EPM156]
    runs_id = ['EPM157','EPM158','EPM151','EPM152','EPM155','EPM156']
    Cs = plt.cm.viridis([0, 0, 0.5, 0.5, 0.8, 0.8])
    LSs = ['-', '--', '-', '--', '-', '--']
    htchs = ["","///","","///","","///"]

    # Function for opening and processing the convective resistance data and storing it in a Pandas dataframe
    def open_processed_data_vol_flux(run, run_id):
        da = run['vol_flux_irminger_srfc'] + run['vol_flux_irminger_depth']
        df = da.to_dataframe(run_id)
        df = df.reset_index()
        df = df.drop('time_centered', axis=1)
        df['time_counter'] = df['time_counter'].astype(str)
        df['time_counter'] = df['time_counter'].map(lambda date_string: datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S'))
        df['time_counter'] = pd.to_datetime(df['time_counter'])
        return df

    # Function for merging dataframes from each run
    def merge_dfs(dfs):
        merged = reduce(lambda  left,right: pd.merge(left,right,on=['time_counter'],how='inner'), dfs)
        return merged
   
    # Opening the convective resistance data
    df = []
    for n,run in enumerate(runs):
        df_temp = open_processed_data_vol_flux(run, runs_id[n])
        df.append(df_temp)
    df = merge_dfs(df)
    df = df.set_index('time_counter')
    df = df.loc['2007-12-01':'2017-11-30'] # Defining our period of interest; note how the years are defined by the winter (i.e., including previous December)
    df = df.where(df!=0) # Masking any spurious zeros 
    df = df/1000000 # Handling the units (J -> PJ)

    # Plotting the convective resistance time series
    df.groupby(df.index.shift(1,freq='m').shift(1,freq='d').year).mean().plot(ax=ax1,xticks=[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],color=Cs,style=LSs,legend=False,zorder=100) #rot=30
    ax1.set_xlabel(None)
    ax1.xaxis.grid(True,linewidth=0.25)
    means = df.mean(axis=0)
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_tick_params(labelsize=9)
    ax1.set_ylabel(r'Volume flux ($Sv$)',fontdict={'fontsize':12})
    ax1.set_title('Volume flux\nWest Greenland Curren → interior Labrador Sea',fontdict={'fontsize':12}) # $\overline{ \int \int _A \Omega_{CR} dA }_{winter}$
    ax1.yaxis.grid(True,linewidth=0.25,zorder=-10)

    # ax2
    da = EPM157['vol_flux_section_mean']/areas.values + EPM151['vol_flux_section_mean']/areas.values + EPM155['vol_flux_section_mean']/areas.values 
    da.plot.pcolormesh(x='ids',y='deptht',rasterized=True, ax=ax2, add_colorbar=False, cmap='BrBG')
    ax2.vlines([66,121,181,231],0,3000,'k')
    ax2.set_ylim(0,3000)
    ax2.invert_yaxis()
    ax2.set_ylabel("Depth ($m$)",fontdict={'fontsize':12})
    ax2.set_title('Volume flux into the interior Labrador Sea\nMean of CGRF simulations',fontdict={'fontsize':12})
    ax2.set_xlabel("Longitude") 
    ax2.set_xticklabels(EPM151['nav_lon_grid_T'].values)
    ax2.text(0.28,0.075, 'WGC', transform=ax2.transAxes,fontsize=12)
    ax2.text(0.73,0.075, 'Labrador\nCurrent', transform=ax2.transAxes,fontsize=12)

    # ax3
    da = EPM158['vol_flux_section_mean']/areas.values + EPM152['vol_flux_section_mean']/areas.values + EPM156['vol_flux_section_mean']/areas.values
    da.plot.pcolormesh(x='ids',y='deptht',rasterized=True, ax=ax3, add_colorbar=False, cmap='BrBG')
    ax3.vlines([66,121,181,231],0,3000,'k')
    ax3.set_ylim(0,3000)
    ax3.invert_yaxis()
    ax3.set_ylabel("Depth ($m$)",fontdict={'fontsize':12})
    ax3.set_title('Volume flux into the interior Labrador Sea\nMean of ERA-I simulations',fontdict={'fontsize':12})
    ax3.set_xlabel("Longitude")
    ax3.text(0.28,0.075, 'WGC', transform=ax3.transAxes,fontsize=12)
    ax3.text(0.73,0.075, 'Labrador\nCurrent', transform=ax3.transAxes,fontsize=12)
    ax3.set_xticklabels(EPM151['nav_lon_grid_T'].values)
    
    plt.savefig('figure_ls3k_vol_fluxes.pdf',format='pdf',dpi=600)

    
if __name__=="__main__":
    #create_temporary_files()
    #ls3k_plot_fluxes()
    #ls3k_plot_barh_diffs()
    #test_plot_ls3k_region_defs()
    ls3k_plot_volume_fluxes()
