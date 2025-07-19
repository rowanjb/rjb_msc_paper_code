#identifies and removes gridT, gridU, and gridV files that give "all nan slice" errors
#it's slow but you only need to run it once
#Rowan Brown
#May 8, 2023

import xarray as xr
import os

def filepaths(run): 
    """Identifies and removes gridT, gridU, and gridV files that give "all nan slice" errors.
    Sometimes--rarely--the files get corrupted! Esp. true after Graham (HPC) crashed in early '24.
    Filepaths are saved in text files for security reasons.
    All model output files here refer to copies on Graham.
    Run should be a string e.g., 'EPM151' or 'EPM155' etc."""

    # Directory of filepath txt files
    fp_dir = '../filepaths/'

    # Directory of ANHA4 output files
    with open(fp_dir + 'ANHA4_graham_output.txt') as f: lines = f.readlines()
    ANHA4_graham_output_dir = [line.strip() for line in lines][0] 

    # Directory of the run's nemo output files on graham
    nemo_output_dir = ANHA4_graham_output_dir + '/ANHA4-' + run + '-S/'

    # List of filepaths
    filepaths_gridT = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('gridT.nc')])
    filepaths_gridU = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('gridU.nc')])
    filepaths_gridV = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('gridV.nc')])
    filepaths_gridB = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('gridB.nc')])
    filepaths_gridW = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('gridW.nc')])
    filepaths_icebergs = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('icebergs.nc')])
    filepaths_icemod = sorted([nemo_output_dir + file for file in os.listdir(nemo_output_dir) if file.endswith('icemod.nc')])

    # Testing if gridT files are read-able
    bad_files = [] # Initializing list of bad filepaths
    for filepath in filepaths_gridT:
        try:
            DS = xr.open_dataset(filepath)
        except:
            bad_files.append(filepath[:-8]) # Saving any bad filepaths
            print('gridT: ' + filepath)
    
    # Testing if gridU files are read-able
    for filepath in filepaths_gridU:
        try:
            DS = xr.open_dataset(filepath)
        except:
            bad_files.append(filepath[:-8]) # Saving any bad filepaths
            print('gridU: ' + filepath)
    
    # Testing if gridV files are read-able
    for filepath in filepaths_gridV:
        try:
            DS = xr.open_dataset(filepath)
        except:
            bad_files.append(filepath[:-8]) # Saving any bad filepaths
            print('gridV: ' + filepath)
    
    # Testing if gridV files are read-able
    for filepath in filepaths_gridW:
        try:
            DS = xr.open_dataset(filepath)
        except:
            bad_files.append(filepath[:-8]) # Saving any bad filepaths
            print('gridW: ' + filepath)

    # Testing if icemod files are read-able
    for filepath in filepaths_icemod:
        try:
            DS = xr.open_dataset(filepath)
        except:
            bad_files.append(filepath[:-9]) # Saving any bad filepaths
            print('icemod: ' + filepath) 

    # Testing if iceberg files are read-able
    for filepath in filepaths_icebergs:
        try:
            DS = xr.open_dataset(filepath)
        except:
            bad_files.append(filepath[:-11]) # Saving any bad filepaths
            print('icebergs: ' + filepath)

    # Removing duplicates from the list
    bad_files = list( dict.fromkeys(bad_files) )

    # Removing bad filepaths
    for bad_file in bad_files:
        print(bad_file + ' is a bad file')
        filepaths_gridT.remove(bad_file + 'gridT.nc')
        filepaths_gridU.remove(bad_file + 'gridU.nc')
        filepaths_gridV.remove(bad_file + 'gridV.nc')
        filepaths_gridB.remove(bad_file + 'gridB.nc')
        filepaths_gridW.remove(bad_file + 'gridW.nc')
        filepaths_icebergs.remove(bad_file + 'icebergs.nc')
        filepaths_icemod.remove(bad_file + 'icemod.nc')

    # Saving the filepaths as txt files
    with open(fp_dir + run + '_gridT_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_gridT:
            output.write(str(i) + '\n')
    with open(fp_dir + run + '_gridU_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_gridU:
            output.write(str(i) + '\n')
    with open(fp_dir + run + '_gridV_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_gridV:
            output.write(str(i) + '\n')
    with open(fp_dir + run + '_gridB_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_gridB:
            output.write(str(i) + '\n')
    with open(fp_dir + run + '_gridW_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_gridW:
            output.write(str(i) + '\n')
    with open(fp_dir + run + '_icebergs_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_icebergs:
            output.write(str(i) + '\n')
    with open(fp_dir + run + '_icemod_filepaths_may2024.txt', 'w') as output:
        for i in filepaths_icemod:
            output.write(str(i) + '\n')

if __name__ == '__main__':
    for run in ['EPM151','EPM152','EPM155','EPM156','EPM157','EPM158']:
        filepaths(run)
