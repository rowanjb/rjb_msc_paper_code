#Looks at oxygen and CO2 fluxes
#Rowan Brown
#25 Oct 2023

#Try: 12gb and 36hrs

import numpy as np 
import xarray as xr
from datetime import datetime
import os

def bling(run,mask_choice,tracer):
    print(run+' '+mask_choice+' '+tracer)

    #== creating directory if doesn't already exist ==#
    dir = run + '_bling/'
    if not os.path.exists(dir):
        os.makedirs(dir)
   
    #== opening .nc output files ==#
    gridT_txt = run + '_filepaths/' + run + '_gridT_filepaths_bling.txt' #.txt of .nc filepaths
    with open(gridT_txt) as f: lines = f.readlines()
    filepaths_gridT = [line.strip() for line in lines]
    filepaths_gridT = [fp for fp in filepaths_gridT if (datetime(2008,1,1) <= datetime.strptime(fp[-20:-9],'y%Ym%md%d') <= datetime(2018,12,31))]
    preprocess_gridT = lambda ds: ds[['e3t',tracer]] #looking mainly at O2 and CO2
    DST = xr.open_mfdataset(filepaths_gridT[10:15],preprocess=preprocess_gridT)#,engine="netcdf4")
    DST = DST.rename({'deptht': 'z', 'y_grid_T': 'y', 'x_grid_T': 'x'})

    #vooxy {'long_name': 'Dissolved Oxygen Concentration from BLING', 'units': 'mol/m3', 'online_operation': 'average', 'interval_operation': '1080 s', 'interval_write': '5 d', 'cell_methods': 'time: mean (interval: 1080 s)'}
    #vodic {'long_name': 'Dissolved Carbon Concentration from BLING', 'units': 'mol/m3', 'online_operation': 'average', 'interval_operation': '1080 s', 'interval_write': '5 d', 'cell_methods': 'time: mean (interval: 1080 s)'}

    #== gridT mask and cell dims. ==#
    with xr.open_dataset('masks/ANHA4_mesh_mask.nc') as DS:
        DST = DST.assign_coords(tmask=DS.tmask[0,:,:,:])
        DST['e1t'] = DS.e1t[0,:,:] 
        DST['e2t'] = DS.e2t[0,:,:]

    #== masking ==#
    if mask_choice == 'LS2k': 
        with xr.open_dataarray('masks/mask_LS_2k.nc') as DS:
            DST = DST.assign_coords(mask=DS[:,:,0].astype(int).rename({'x_grid_T':'x','y_grid_T':'y'}))
    elif mask_choice == 'LS': 
        with xr.open_dataarray('masks/mask_LS.nc') as DS:
            DST = DST.assign_coords(mask=DS[:,:,0].astype(int).rename({'x_grid_T':'x','y_grid_T':'y'}))
    elif mask_choice == 'LSCR': 
        with xr.open_dataset('masks/ARGOProfiles_mask.nc') as DS:
            DST = DST.assign_coords(mask = DS.tmask.astype(int))
    elif mask_choice == 'LS3k':
        with xr.open_dataarray('masks/mask_LS_3000.nc') as DS:
            DST = DST.assign_coords(mask=DS[:,:,0].astype(int).rename({'x_grid_T':'x','y_grid_T':'y'}))
    else:
        print("Y'all didn't choose a mask")
        quit()
    DST = DST.where(DST.mask == 1, drop=True)

    print(DST['vodic'].isel(time_counter=2, y=30, x=25).to_numpy())
    quit()

    #== total concentration in each cell ==#
    DST['volumes'] = DST.e1t*DST.e2t*DST.e3t
    DST['amount_in_each_cell'] = DST[tracer]*DST['volumes'] #tracer is in mol/m3 

    #== considering total concentrations to different depths ==#
    for d in [200]:#, 1000, 200, 50]: #loop through the 4 depths  

        DSTd = DST.where(DST.z < d, drop=True) #drop values below specified depth

        ##masking shelves
        #I DON'T THINK I NEED TO MASK THE SHELVES HERE SINCE WE'RE LOOKING AT TOTAL CONCENTRATION, NOT AVG. CONCENTRATION
        ##NOTE: bathy is masked to avoid skewed understandings/results from the on-shelf values this section could be commented out if needed 
        #bottom_slice = DS_d.votemper.isel(deptht = -1).isel(time_counter = 0)
        #bottom_slice_bool = bottom_slice.notnull()
        #shelf_mask, temp = xr.broadcast(bottom_slice_bool, DS_d.votemper.isel(time_counter=0))
        #DS_d = DS_d.where(shelf_mask)

        #== final calcs. and saving ==#
        DSTd['avg_concentration_over_time'] = DST['amount_in_each_cell'].sum(dim=['x','y','z'])/DST['volumes'].sum(dim=['x','y','z']) #i.e., total amount div by total vol
        #DSTd['concentration_per_column'] = DSTd.concentration_in_each_cell.sum(dim=['z']).mean(dim=['time_counter'])
        DSTd['avg_concentration_over_time'].to_netcdf(run + '_bling/' + run + '_' + tracer + '_timeplot_' + mask_choice + str(d) + '.nc')
        #DSTd.concentration_per_column.to_netcdf(run + '_bling/' + run + '_' + tracer + '_map_' + mask_choice + str(d) + '.nc')

    DST.close()

if __name__ == '__main__':
    for i in ['LS3k']:# ['LS','LS2k','LSCR']:
        for j in ['vodic']:
            for r in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
                bling(run=r,mask_choice=i,tracer=j) #'vooxy' or 'vodic'
        #air(run=i,mask_choice='LS')
        #for j in ['LS2k','LS','LSCR']:
            #air(run=i,mask_choice=j)

