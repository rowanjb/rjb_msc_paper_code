# For creating figure showing the ANHA4 grid
# Rowan Brown
# July 2025 

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import numpy as np
import scipy

# ds containing the grid sizes
mesh_mask = 'masks/ANHA4_mesh_mask.nc'
ds = xr.open_dataset(mesh_mask)

# Open example file
gridT_txt = '../filepaths/EPM151_gridT_filepaths_bling.txt' 
with open(gridT_txt) as f: lines = f.readlines()
filepaths_gridT = [line.strip() for line in lines]
ds_ex = xr.open_dataset(filepaths_gridT[10])
mask = ds_ex['votemper'].isel(deptht=0,time_counter=0).to_numpy()

# Combine
ds = ds.assign_coords({'mask': (['y','x'],mask)})# ds_ex.isel(depth=0,time_counter=0)#.astype('bool')

# Masking the empty cells and converting to km
ds['e1t'] = xr.where(~np.isnan(ds['mask']),ds['e1t'].isel(t=0)/1000,ds['mask'])

# Shapefile of land with 1:50,000,000 scale
land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='gray')

# Defining the projection, note that standard parallels are the parallels of correct scale
projection = ccrs.NearsidePerspective(central_longitude=-35,  central_latitude=35, satellite_height=10005785831)

# Create figure (using the specified projection)
ax = plt.subplot(1, 1, 1, projection=projection)
cm = 1/2.54  # centimeters in inches
plt.gcf().set_figwidth(9*cm)

# This is to ensure you don't clip any edge of the circle, which just looks ugly
# from https://stackoverflow.com/questions/73178207/how-to-plot-a-map-of-a-semi-sphere-eg-northern-hemisphere-using-matplotlib-car
r = 6378000 #radius of earth in m
ax.set_xlim(-r, r)
ax.set_ylim(-r, r)

# Colour map; good for everyone
cm = 'viridis'

# Plotting data
ax.set_global()
ax.stock_img()
p1 = ax.pcolormesh(ds.nav_lon, ds.nav_lat, ds.e1t, transform=ccrs.PlateCarree(), cmap=cm) 
ax_cb = plt.axes([0.15, 0.15, 0.7, 0.022])
cb = plt.colorbar(p1,cax=ax_cb,orientation='horizontal')
cb.set_label(label='Grid size ($km$)', fontsize=12)
cb.ax.tick_params(labelsize=9)

# Add coast lines 
ax.coastlines(resolution='10m', linewidth=0.5)

# Aave and close figure
plt.savefig('figure_ANHA4_grid.png', dpi=600, bbox_inches="tight")
#plt.savefig('ANHA4_domain_map.pdf', format='pdf', bbox_inches="tight")
plt.clf()
