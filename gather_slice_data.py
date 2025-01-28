import numpy as np
import pandas as pd
import glob
import numpy as np
from read_tecplot_data import *
from visualize_wing import *
import pickle
import sys
import argparse

def main():
    # Parse COmmand Line Inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', required=True, help='Path to output file')
    parser.add_argument('--data_dirs', required=True, help='Path to raw tecplot data files')
    parser.add_argument('--data_type', required=True, help='aero data type')
    args = parser.parse_args()
    
    # Define data structure to store raw wing data
    dataframes = {}

    # Get and sort data folders
    folders = glob.glob(args.data_dirs+'case*')
    sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1]))

    # Loop over cases
    for folder in sorted_folders:
        # Identify case we are currently extracting
        case = folder.split('_')[-1]
        
        # Temporary data structures
        case_data = []
        tmp_dict = {}

        # Get and store slices for given case
        files = glob.glob(folder+'/*_slices.dat')
        sorted_files = sorted(files, key=lambda x: int(x.split('fc_')[1].split('_')[0]))
        
        # Loop over slices
        for file in sorted_files:
            print(file)

            # Read data file
            if args.data_type == '2D':
                data = [read_2d_slice_data(file)]
            elif args.data_type == '3D':
                data = read_3d_slice_data(file)
            else:
                sys.exit('Unsupported Data Type (Must be 2D airfoil or 3D wing)')

            # Get leading edge point of most root-wise cross section
            x, y = data[0].values[:,0], data[0].values[:,1]
            imin = np.argmin(x)
            x0, y0 = x[imin], y[imin]
            
            # Uniformly shift x, y coordinates so that leading eadge of my root-wise cross section lies at (0,0)
            for df in data:
                df['CoordinateX'] -= x0
                df['CoordinateY'] -= y0

            slices = file.split('/')[-1].split('_')[1]

            # Ordering of most spanwise coordinates buggy so let's ignore for now (will fuck up spline fit later on)
            if args.data_type == '3D':
                tmp_dict[slices] = data[:-1]
                data = data[:-1]
            else: 
                tmp_dict[slices] = data

            fname = 'images/'+args.data_type+'/'+case+'_'+slices+'.png'
            plot_2D_airfoil_section(data, fname)

        dataframes[case] = tmp_dict

    with open(args.output_file, 'wb') as handle:
        pickle.dump(dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()