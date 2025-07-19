# Specifies the interior Lab Sea bounding cells along the 3,000 m isobath boundary 
# Rowan Brown
# July, 2025

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature

def ls3k_boundary():
    """Specifies the Lab Sea interior bounding cells along the 3,000 m isobath.
    (i.e., the outermost cells in the mask.)
    Copied from an earlier ad hoc script.
    Extremely ugly algorithm. It works, though."""

    mask_fp = 'masks/mask_LS_3000.nc'
    xrData = xr.open_dataset(mask_fp).isel(deptht=0).drop_vars('deptht')
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

    return z3

def test_plot_ls3k_boundary():
    z = ls3k_boundary()
    mask_fp = 'masks/mask_LS_3000.nc'
    xrData = xr.open_dataset(mask_fp).isel(deptht=0).drop_vars('deptht')

    westLon = -65
    eastLon = -40
    northLat = 67
    southLat = 51

    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='gray')
    projection = ccrs.AlbersEqualArea(central_longitude=-55, central_latitude=50,standard_parallels=(southLat,northLat))
    ax = plt.subplot(1, 1, 1, projection=projection)
    ax.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.coastlines(resolution='50m')
    ax.pcolormesh(xrData['nav_lon_grid_T'], xrData['nav_lat_grid_T'], z, transform=ccrs.PlateCarree())
    plt.savefig('test.png')

if __name__=="__main__":
    test_plot_ls3k_boundary()
