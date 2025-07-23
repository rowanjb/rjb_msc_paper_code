# For creating background context figures
# Will annotate using inkscape
# Rowan Brown 
# July 2025

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from scipy import interpolate
import numpy as np
import os
import PIL.Image

# ds containing the grid sizes
mesh_mask = 'masks/ANHA4_mesh_mask.nc'
ds = xr.open_dataset(mesh_mask)

# Open example file
gridT_txt = '../filepaths/EPM151_gridT_filepaths.txt'
with open(gridT_txt) as f: lines = f.readlines()
filepaths_gridT = [line.strip() for line in lines]
ds_ex = xr.open_dataset(filepaths_gridT[10])
mask = ds_ex['votemper'].isel(deptht=0,time_counter=0).to_numpy()

# Combine
ds = ds.assign_coords({'mask': (['y','x'],mask)})

# Masking the empty cells and converting to km
ds['e1t'] = xr.where(~np.isnan(ds['mask']),ds['e1t'].isel(t=0)/1000,ds['mask'])

# Shapefile of land with 1:50,000,000 scale
land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='gray')

# Defining the projection, note that standard parallels are the parallels of correct scale
projection = ccrs.NearsidePerspective(central_longitude=-35,  central_latitude=35, satellite_height=10005785831)

# Create figure (using the specified projection)
cm = 1/2.54
layout = [['ax1','ax2'],
          ['ax1','ax2'],
          ['ax1','ax2'],
          ['ax1','.']]
fig, axd = plt.subplot_mosaic(layout, figsize=(19*cm, 11*cm), subplot_kw=dict(projection=projection))
ax1 = axd['ax1']
ax2 = axd['ax2']

# This is to ensure you don't clip any edge of the circle, which just looks ugly
# from https://stackoverflow.com/questions/73178207/how-to-plot-a-map-of-a-semi-sphere-eg-northern-hemisphere-using-matplotlib-car
r = 6378000 #radius of earth in m
ax2.set_xlim(-r, r)
ax2.set_ylim(-r, r)

# Colour map; good for everyone
cm = 'viridis'

# Plotting data
ax2.set_global()
ax2.stock_img()
p1 = ax2.pcolormesh(ds.nav_lon, ds.nav_lat, ds.e1t, transform=ccrs.PlateCarree(), cmap=cm, rasterized=True)
ax_cb = plt.axes([0.55, 0.2, 0.35, 0.022])
cb = plt.colorbar(p1,cax=ax_cb,orientation='horizontal')
cb.set_label(label='Grid size ($km$)', fontsize=12)
cb.ax.tick_params(labelsize=9)

# Add coast lines
ax2.coastlines(resolution='10m', linewidth=0.5)

# Finally, add the label
ax2.text(0.1, 0.935, 'b', transform=ax2.transAxes,fontsize=14, fontweight='bold', va='top', ha='right',bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.1'))

#-- set environment variable CARTOPY_USER_BACKGROUNDS
# from: https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-example-high-resolution-background-image-plot.html
os.environ['CARTOPY_USER_BACKGROUNDS'] = '../Backgrounds/'

# Size of the map
westLon = -63
eastLon = -43
northLat = 70
southLat = 50

# Shapefile of land with 1:50,000,000 scale
land = feature.NaturalEarthFeature('physical', 'land', '10m')

# Defining the projection, note that standard parallels are the parallels of correct scale
projection = ccrs.AlbersEqualArea(central_longitude=-55, central_latitude=50,standard_parallels=(southLat,northLat))

plt.switch_backend('agg') # from DKRZ guide

# Create figure (using the specified projection)
ax1.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
c = colors.to_rgba('white', alpha=.7)

# Ticks
gl = ax1.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.5)
gl.top_labels=False #suppress top labels
gl.right_labels=False #suppress right labels
gl.rotate_labels=False
gl.ylocator = mticker.FixedLocator([51, 54, 57, 60, 63, 66, 69, 72])
gl.xlocator = mticker.FixedLocator([-35, -40, -45, -50, -55, -60, -65, -70, -75, -80]) 
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}
gl.xlines = False
gl.ylines = False

# Add land to map
#PIL.Image.MAX_IMAGE_PIXELS = 233280001
#ax1.background_img(name='NaturalEarthRelief', resolution='high')
ax1.stock_img()

# Add coast lines 
ax1.coastlines(resolution='10m', linewidth=0.5)

# Colourmap
cmap = plt.cm.Greys_r
cmap.set_bad(color='none') 

# Plotting data
mask_fp = 'masks/mask_LS_3000.nc'
xrData = xr.open_dataset(mask_fp).isel(deptht=0).drop_vars('deptht')

# Ugly but quick way to define the mask's bounding cells using matrix math
z = xrData.mask_LS_3000.to_numpy().astype(float)
z2 = xrData.mask_LS_3000.to_numpy().astype(float)
z3 = z2.copy()
for row in range(np.shape(z2)[0])[1:-1]:
    for col in range(np.shape(z2)[1])[1:-1]:
        if z2[row,col]==1:
            if z2[row-1,col] == 1: z3[row-1,col] = z3[row-1,col] + 1
            if z2[row+1,col] == 1: z3[row+1,col] = z3[row+1,col] + 1
            if z2[row,col-1] == 1: z3[row,col-1] = z3[row,col-1] + 1
            if z2[row,col+1] == 1: z3[row,col+1] = z3[row,col+1] + 1
            if z2[row+1,col+1] == 1: z3[row+1,col+1] = z3[row+1,col+1] + 1
            if z2[row+1,col-1] == 1: z3[row+1,col-1] = z3[row+1,col-1] + 1
            if z2[row-1,col+1] == 1: z3[row-1,col+1] = z3[row-1,col+1] + 1
            if z2[row-1,col-1] == 1: z3[row-1,col-1] = z3[row-1,col-1] + 1
            else: z3[row,col] = 0
z3[z3==9] = 0
z3[z3>0] = 1
z3[z3==0] = np.nan

# Plotting the bounding cells
p1 = ax1.pcolormesh(ds.nav_lon, ds.nav_lat, z3, transform=ccrs.PlateCarree(), cmap=cmap, rasterized=True)

# Finally, add the label
ax1.text(0.1, 0.95, 'a', transform=ax1.transAxes,fontsize=14, fontweight='bold', va='top', ha='right',bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.1'))

# Save and close figure (generally prefer pdf, but my inkscape crashes with pdfs)
#plt.savefig('figure_LabSea_and_grid_map.pdf', format='pdf', dpi=600) 
plt.savefig('figure_LabSea_and_grid_map.jpg', dpi=1200) 
plt.clf()
