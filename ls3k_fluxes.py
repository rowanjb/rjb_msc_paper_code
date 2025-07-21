# For calculating volume, salt, and heat flux into the interior Lab Sea across the 3,000 m isobath
# Scripts should be fairly easy to generalise for other NEMO or GCM simulations
# Rowan Brown
# July, 2025

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
from ls3k_mask_boundary import ls3k_boundary
import gsw

def identify_flux_faces():
    """Identifies the U and V faces defining the edge of the 3,000 m isobath mask.
    Note that the NEMO grid defines U on the east face of a cell and V on the north face.
    Also defines the inward direction."""

    mask_fp = 'masks/mask_LS_3000.nc'
    mesh_fp = 'masks/ANHA4_mesh_mask.nc'
    ds = xr.open_dataset(mask_fp).isel(deptht=0).drop_vars('deptht')
    ds_mesh = xr.open_dataset(mesh_fp)

    # Pseudo code
    # Loop through each cell
    #   if in mask, then
    #       if empty to right -> save this U face id in a westward-defined-as-inward array (right to left) 
    #       if empty to left -> save U face id minus 1 in an eastward-defined-as-inward array
    #       if empty to bottom -> save V face id minus 1 in a northward-defined-as-inward array
    #       if empty to top -> save V face id in a southward-defined-as-inward
    # Note that we don't use elif in case there is a 1-cell thick peninsula 
    # (Although I doubt anyone will ever read this...)

    rows, cols = np.shape(ds['mask_LS_3000'])
    westward_cells = np.zeros((rows,cols))
    eastward_cells = np.zeros((rows,cols))
    northward_cells = np.zeros((rows,cols))
    southward_cells = np.zeros((rows,cols))

    for row in np.arange(1,rows-1):
        for col in np.arange(1,cols-1):
            if ds['mask_LS_3000'].isel(y_grid_T=row,x_grid_T=col): 
                if ds['mask_LS_3000'].isel(y_grid_T=row,x_grid_T=col+1)==False:
                    westward_cells[row,col] = 1
                if ds['mask_LS_3000'].isel(y_grid_T=row,x_grid_T=col-1)==False:
                    eastward_cells[row,col-1] = 1
                if ds['mask_LS_3000'].isel(y_grid_T=row+1,x_grid_T=col)==False:
                    southward_cells[row,col] = 1
                if ds['mask_LS_3000'].isel(y_grid_T=row-1,x_grid_T=col)==False:
                    northward_cells[row-1,col] = 1

    ds = ds.assign(westward_cells=(('y_grid_T','x_grid_T'),westward_cells))
    ds = ds.assign(eastward_cells=(('y_grid_T','x_grid_T'),eastward_cells))
    ds = ds.assign(northward_cells=(('y_grid_T','x_grid_T'),northward_cells))
    ds = ds.assign(southward_cells=(('y_grid_T','x_grid_T'),southward_cells))

    ds.to_netcdf('masks/ls3k_flux_mask.nc')

def test_plot_ls3k_flux_boundary():
    """Throw-away script showing the success of the identify_flux_faces function."""

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
    plt.savefig('test.png',dpi=1200)

def linearise_flux_face_ids():
    """Create list of y and x ids for all flux faces.
    Importantly, they should be in order such that the region is closed and Hovmoeller diagrams 
    can be plotted with ease. Has double benefit of probably making the flux calculations relatively
    quick. Also note the direction."""

    # Pseudo code
    # Imagine trying to traverse the boundary of a closed region in a CCW direction
    # Starting from a random northward-defined-as-inward flux face cell...
    # If you're at a northward-defined-as-inward cell
    #   Is the cell at row,col+1 northward -> go there
    #   Is the cell at row,col eastward -> go there
    #   Is the cell at row+1,col westward -> go there
    # If you're at a westward-defined-as-inward cell
    #   Is the cell at row+1,col westward -> go there
    #   Is the cell at row,col souththward -> go there
    #   Is the cell at row,col+1 northward -> go there
    # If you're at a southward-defined-as-inward cell
    #   Is the cell at row,col-1 southward -> go there
    #   Is the cell at row,col-1 eastward -> go there
    #   Is the cell at row+1,col-1 westward -> go there
    # If you're at a eastward-defined-as-inward cell
    #   Is the cell at row-1,col eastward -> go there
    #   Is the cell at row-1,col+1 northward -> go there
    #   Is the cell at row-1,col southward -> go there

    def northward(row,col):
        if northward_cells[row,col+1]==1: 
            return row,col+1,'northward'
        elif eastward_cells[row,col]==1:
            return row,col,'eastward'
        elif westward_cells[row+1,col]==1:
            return row+1,col,'westward'
        else:
            print('Broken loop code 1')
            quit()

    def westward(row,col):
        if westward_cells[row+1,col]==1:
            return row+1,col,'westward'
        elif southward_cells[row,col]==1:
            return row,col,'southward'
        elif northward_cells[row,col+1]==1:
            return row,col+1,'northward'
        else:
            print('Broken loop code 2')
            quit()

    def southward(row,col):
        if southward_cells[row,col-1]==1:
            return row,col-1,'southward'
        elif eastward_cells[row,col-1]==1:
            return row,col-1,'eastward'
        elif westward_cells[row+1,col-1]==1:
            return row+1,col-1,'westward'
        else:
            print('Broken loop code 3')
            quit()

    def eastward(row,col):
        if eastward_cells[row-1,col]==1:
            return row-1,col,'eastward'
        elif northward_cells[row-1,col+1]==1:
            return row-1,col+1,'northward'
        elif southward_cells[row-1,col]==1:
            return row-1,col,'southward'
        else:
            print('Broken loop code 4')
            quit()

    ds = xr.open_dataset('masks/ls3k_flux_mask.nc')
    westward_cells = ds['westward_cells'].to_numpy()
    eastward_cells = ds['eastward_cells'].to_numpy()
    northward_cells= ds['northward_cells'].to_numpy()
    southward_cells= ds['southward_cells'].to_numpy()

    ids_of_northward_cells = np.where(northward_cells==1)
    row_of_first_northward_cell, col_of_first_northward_cell = ids_of_northward_cells[0][0], ids_of_northward_cells[1][0]
    row_of_second_northward_cell, col_of_second_northward_cell = ids_of_northward_cells[0][1], ids_of_northward_cells[1][1]

    final_rows, final_cols, final_directions = [], [], []
    row, col = row_of_second_northward_cell, col_of_second_northward_cell 
    direction = 'northward'
    final_rows.append(row), final_cols.append(col), final_directions.append(direction)
    print(row, col, direction)
    while row!=row_of_first_northward_cell or col!=col_of_first_northward_cell:
        if direction=='northward':
            row, col, direction = northward(row,col)
        elif direction=='westward':
            row, col, direction = westward(row,col)
        elif direction=='southward':
            row, col, direction = southward(row,col)
        elif direction=='eastward':
            row, col, direction = eastward(row,col)
        else:
            print('Broken loop code 5')
            quit()
        final_rows.append(row), final_cols.append(col), final_directions.append(direction)
        print(row, col, direction)

    ids = np.arange(len(final_rows))
    ds_final = xr.Dataset(
        {
            "y_grid_T": (("ids"),final_rows),
            "x_grid_T": (("ids"),final_cols),
            "directions": (("ids"),final_directions),
        },
        coords={
            "ids": ids,
        },
        attrs=dict(description="Indices and directions of all faces which have non-zero flux into the interior of a masked area"),
    )
    ds_final.to_netcdf('masks/ls3k_flux_face_ids.nc')

def test_plot_ls3k_flux_boundary_faces():
    """Throw-away script showing the success of the linearise_flux_face_ids function.
    If you save it at a high enough dpi, it shows how we programmatically identified the 
    coordinates of the faces bounding a closed region, and the inwards-positive orientation 
    of the faces."""
    
    # Open the data and various mask and mesh files
    ds = xr.open_dataset('masks/ls3k_flux_face_ids.nc')
    mask_fp = 'masks/mask_LS_3000.nc'
    mesh_fp = 'masks/ANHA4_mesh_mask.nc'
    ds_mask = xr.open_dataset(mask_fp).isel(deptht=0).drop_vars('deptht')
    ds_mesh = xr.open_dataset(mesh_fp)

    # Need to open some example grid U and grid V files for their gridU and gridV lons and lats
    with open('../filepaths/EPM155_gridU_filepaths.txt') as f: lines = f.readlines()
    example_gridU_fp = [line.strip() for line in lines][10]
    with open('../filepaths/EPM155_gridV_filepaths.txt') as f: lines = f.readlines()
    example_gridV_fp = [line.strip() for line in lines][10]
    dsu = xr.open_dataset(example_gridU_fp)
    dsv = xr.open_dataset(example_gridV_fp)

    rows = ds['y_grid_T'].to_numpy()
    cols = ds['x_grid_T'].to_numpy()
    directions = ds['directions'].to_numpy()
    lons, lats = [], []
    u, v = [], []
    for n,row in enumerate(rows):
        col = cols[n]
        if directions[n]=='northward':
            lons.append(dsv.nav_lon.isel(y=row,x=col).to_numpy())
            lats.append(dsv.nav_lat.isel(y=row,x=col).to_numpy())
            v.append(0.3),u.append(0)
        if directions[n]=='southward':
            lons.append(dsv.nav_lon.isel(y=row,x=col).to_numpy())
            lats.append(dsv.nav_lat.isel(y=row,x=col).to_numpy())
            v.append(-0.3),u.append(0)
        if directions[n]=='eastward':
            lons.append(dsu.nav_lon.isel(y=row,x=col).to_numpy())
            lats.append(dsu.nav_lat.isel(y=row,x=col).to_numpy())
            u.append(0.3),v.append(0)
        if directions[n]=='westward':
            lons.append(dsu.nav_lon.isel(y=row,x=col).to_numpy())
            lats.append(dsu.nav_lat.isel(y=row,x=col).to_numpy())
            u.append(-0.3),v.append(0)
    
    lons = [float(i) for i in lons]
    lats = [float(i) for i in lats]

    westLon, eastLon, northLat, southLat = -65, -40, 67, 51
    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='gray')
    projection = ccrs.AlbersEqualArea(central_longitude=-55, central_latitude=50,standard_parallels=(southLat,northLat))
    ax = plt.subplot(1, 1, 1, projection=projection)
    ax.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.coastlines(resolution='50m')
    ax.pcolormesh(ds_mesh.nav_lon,ds_mesh.nav_lat,ds_mask['mask_LS_3000'],transform=ccrs.PlateCarree())
    ax.quiver(np.array(lons),np.array(lats),np.array(u),np.array(v),transform=ccrs.PlateCarree(),width=0.001)
    plt.savefig('test.png', dpi=1200)

def identify_projected_dims():
    """Identifies the horizontal dimensions of the projected areas.
    Previously calculated the total area using invariant e3t values (cell heights).
    Now updated so that calculate_fluxes() can utilise variable cell heights.
    Saves output into a netcdf file."""

    # Open the data and various mask and mesh files
    ds = xr.open_dataset('masks/ls3k_flux_face_ids.nc')
    ds_mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc').isel(t=0)    

    # Calculating the projected areas
    ds_mesh['U_face_areas'] = ds_mesh['e2u']#*ds_mesh['e3u'] # 2 == north-west dim
    ds_mesh['V_face_areas'] = ds_mesh['e1v']#*ds_mesh['e3v'] # 1 == east-west dim
   
    # Constructing array of areas
    num_depths = len(ds_mesh['z'].to_numpy())
    num_cells = len(ds['ids'].to_numpy())
    rows = ds['y_grid_T'].values
    cols = ds['x_grid_T'].values
    directions = ds['directions'].values
    areas = np.zeros( (num_cells, num_depths) )
    for n in np.arange(num_cells): # For each cell in the boundary 
        row, col, direction = rows[n], cols[n], directions[n]
        if direction=='southward' or direction=='northward': # If the flux is north-south... 
            areas[n,:] = ds_mesh['V_face_areas'].isel(y=row,x=col).values # Then we want a v dimension
        elif direction=='eastward' or direction=='westward': # If the flux is east-west...
            areas[n,:] = ds_mesh['U_face_areas'].isel(y=row,x=col).values # Then we want a u dimension
        else:
            print('Broken loop with error code 6')
            quit()

    ds_final = xr.Dataset(
        {
            "horiz_dim": (("ids","z"),areas),
        },
        coords={
            "ids": ds['ids'].values,
            "z": ds_mesh['z'].values,
        },
        attrs=dict(description="Appropriate horizontal dimension of all faces which have non-zero flux into the interior of masked area."
                                "If flux is east-west (U velocity) then the dimension is e2u."
                                "If flux is north-south (V velocity) then the dimension is e1v"),
    )
    ds_final.to_netcdf('masks/ls3k_flux_face_hzdims.nc')

def calculate_fluxes(run):
    """Calculates volume, temperature, and salinity fluxes into the interior Lab Sea.
    I am taking the T and S from the same cell coordinates as U and V.
    They might be very slightly more accurate if taken as interpolated values at the points themselves, 
    but I will revisit this if there is an issue with the heat and salt budget.
    Saves output into a netcdf file."""

    # Open the data and various mask and mesh files
    ds_ids = xr.open_dataset('masks/ls3k_flux_face_ids.nc')
    ds_hzdims = xr.open_dataset('masks/ls3k_flux_face_hzdims.nc')
    ds_mesh = xr.open_dataset('masks/ANHA4_mesh_mask.nc')
    ds_mask = xr.open_dataset('masks/mask_LS_3000.nc').isel(deptht=0).drop_vars('deptht')

    # Open the U and V files 
    with open('../filepaths/'+run+'_gridT_filepaths_jul2025.txt') as f: lines = f.readlines()
    gridT_fps = [line.strip() for line in lines]
    with open('../filepaths/'+run+'_gridU_filepaths_jul2025.txt') as f: lines = f.readlines()
    gridU_fps = [line.strip() for line in lines]
    with open('../filepaths/'+run+'_gridV_filepaths_jul2025.txt') as f: lines = f.readlines()
    gridV_fps = [line.strip() for line in lines]
    pp_gridT = lambda ds: ds[['votemper','vosaline']]
    pp_gridU = lambda ds: ds[['e3u','vozocrtx']]
    pp_gridV = lambda ds: ds[['e3v','vomecrty']]
    dst = xr.open_mfdataset(gridT_fps, preprocess=pp_gridT)
    dsu = xr.open_mfdataset(gridU_fps, preprocess=pp_gridU).rename({'depthu':'deptht'}) # (They're equal)
    dsv = xr.open_mfdataset(gridV_fps, preprocess=pp_gridV).rename({'depthv':'deptht'})

    # Extracting only the boundary cells
    rows = ds_ids['y_grid_T'].values
    cols = ds_ids['x_grid_T'].values
    directions = ds_ids['directions'].values
    dst = dst.isel(x_grid_T=xr.DataArray(cols, dims="ids"), y_grid_T=xr.DataArray(rows, dims="ids"))
    dsu = dsu.isel(x=xr.DataArray(cols, dims="ids"), y=xr.DataArray(rows, dims="ids"))
    dsv = dsv.isel(x=xr.DataArray(cols, dims="ids"), y=xr.DataArray(rows, dims="ids"))

    # Gathering necessary terms into the gridT dataset
    # Basic idea is to multiply U and V velocities by direction vectors so that each boundary face (which 
    # has a unique id) has an associated non-zero U or V velocity (depending whether it is an 
    # east-west face (V velocity), or a north-south face (U velocity)). The other component is made
    # equal to zero and then added together. If the inward direction is against the positive 
    # coordinate system of the model, then the non-zero velocity component is also multiplied by -1.
    # The velocity component is also multiplied by the e3 variable, since it will later by multiplied
    # by the appropriate e1 or e2 variable, which will give the volume flux.
    direction_dict_U = {'southward':0,'westward':-1,'eastward':1,'northward':0}
    direction_modulator_U = [direction_dict_U[direction] for direction in directions]
    direction_dict_V = {'southward':-1,'westward':0,'eastward':0,'northward':1}
    direction_modulator_V = [direction_dict_V[direction] for direction in directions]
    dst['vel*e3'] = dsu['e3u']*dsu['vozocrtx']*direction_modulator_U + dsv['e3v']*dsv['vomecrty']*direction_modulator_V
    dst['horiz_dims'] = (("ids","deptht"),ds_hzdims['horiz_dim'].values)

    # Need densitie; uses some minor simplifications
    dst['pressure'] = gsw.p_from_z((-1)*dst['deptht'],dst['nav_lat_grid_T']) # dbar, note deptht needs to be negative
    dst['SA'] = gsw.SA_from_SP(dst['vosaline'],dst['pressure'],dst['nav_lon_grid_T'],dst['nav_lat_grid_T']) # unitless, i.e., g/kg 
    dst['CT'] = gsw.CT_from_pt(dst['SA'],dst['votemper']) # C
    dst['cp'] = gsw.cp_t_exact(dst['SA'],gsw.t_from_CT(dst['SA'],dst['CT'],dst['pressure']),dst['pressure']) # J/kg C
    dst['pot_dens'] = gsw.rho(dst['SA'],dst['CT'],0) # kg/m**3, equal to potential density if pressure = 0
    dst['insit_dens'] = gsw.rho(dst['SA'],dst['CT'],dst['pressure']) # kg/m**3

    # Volume flux 
    dst['vol_flux'] = dst['horiz_dims']*dst['vel*e3'] # m * m**2/s = m**3/s
    
    # Heat flux, calculated with refT=-2, gsw.cp_t_exact, and gsw.pot_dens, which is same as my heat content calcs
    refT = -2
    dst['heat_flux'] = dst['vol_flux']*dst['pot_dens']*dst['cp']*(dst['votemper']-refT) # m**3/s * kg/m**3 * J/kgC * C = J/s

    # Salt content
    dst['salt_flux'] = dst['vol_flux']*dst['vosaline']*dst['insit_dens'] # m**3/s * g/kg * kg/m**3

    # Saving
    dst = dst[['areas','vol_flux','heat_flux','salt_flux']]
    dst = dst.assign_attrs(description="Fluxes into the interior Lab Sea", 
                           title="Fluxes into the interior Lab Sea",
                           vol_flux_units="m**3/s",
                           heat_flux_units="J/s",
                           salt_flux_units="g/s")
    print("LS3k fluxes calc'd for: "+run)
    dst.to_netcdf('ls3k_fluxes_'+run+'.nc')
    dst.close()
    dsu.close()
    dsv.close()
    print("LS3k fluxes saved for: "+run)

def test_ls3k_fluxes():
    """Throw-away script for testing if the fluxes calculated in calculate_fluxes() are reasonable"""

    ds = xr.open_dataset("ls3k_fluxes_EPM151.nc")
    
    print(ds['vol_flux'].sum(['ids','deptht']).mean().values)
    quit()

    # Cape Farwell is at ids=80
    ds['deptht'] = ds['deptht']*(-1)
    da = ds['vol_flux'].mean('time_counter')
    fig, ax = plt.subplots(figsize=(20,4))
    da.plot.pcolormesh(x='ids',y='deptht', ax=ax)
    ax.set_xlabel('Perimeter cell id')
    ax.set_ylabel('Depth ($m$)')
    ax.set_ylim(-4000,0)
    ax.set_title("Mean volume flux EPM151 2002-2018")
    plt.savefig('test.png')

    print(ds)

if __name__=="__main__":
    #identify_flux_faces()
    #test_plot_ls3k_flux_boundary()
    #linearise_flux_face_ids()
    #test_plot_ls3k_flux_boundary_faces()
    #identify_projected_dims()
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
        calculate_fluxes(run)
